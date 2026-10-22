"""Exact density variation of the adopted zero-stochastic-fiducial forecast.

The clustering signal, AP map and its cosmological responses are fixed. Only
the noise-normalized stochastic columns and auto Poisson covariance change.
"""
from copy import deepcopy

import numpy as np

from pfsfog.elf_covariance_audit import covariance_from_spectra
from pfsfog.fisher_core import fisher_from_shells
from scripts.elf_study_data import project_cell


PFS = 'PFS-ELG'
AUTO_STOCH = frozenset(('Pshot', 'a0', 'a2'))
CROSS_STOCH = frozenset(('Pshot_cross', 'a2_cross'))


def _factor(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError('density factor must be a positive finite number')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('density factor must be a positive finite number')
    return value


def density_column_scales(names, factor):
    """Derivative scaling in the existing dimensionless nuisance coordinates."""
    factor = _factor(factor)
    scales = np.ones(len(names))
    for i, name in enumerate(names):
        fields = name.split(':')
        if len(fields) != 3:
            continue
        population, coordinate = fields[1:]
        if coordinate in AUTO_STOCH and population == PFS:
            scales[i] = 1/factor
        elif coordinate in CROSS_STOCH and PFS in population.split('|'):
            if len(population.split('|')) != 2:
                raise ValueError('invalid cross stochastic population label')
            scales[i] = 1/np.sqrt(factor)
    return scales


def gamma3_zero_support(inputs):
    """Eulerian one-loop Gamma3 response is A(k)+B(k) mu^2 before AP.

    The ELF 13 coefficients contain b1*bGamma3 and f*mu^2*bGamma3 only;
    there is no Gamma3 in 22. Unequal-tracer symmetrization preserves this
    angular order. Identity AP and IR-off are essential for the ell=4 zero.
    This changes zero classification, not any response or acceptance limit.
    """
    settings=inputs.get('engine_settings',{})
    alphas=np.asarray(inputs.get('analytic_zero_response',{}).get('alphas',[]))
    if (settings.get('engine')!='elf.EPT' or settings.get('do_irres') is not False
            or alphas.shape!=(2,) or not np.isfinite(alphas).all()
            or not np.all(abs(alphas-1.)<=32*np.finfo(float).eps)):
        return []
    result=[]
    for name in inputs['metadata']['local_names']:
        fields=name.split(':')
        if len(fields)==3 and fields[2]=='bGamma3':
            result.extend((name,tuple(pair),4) for pair in inputs['metadata']['retained_pairs']
                          if fields[1] in pair)
    return result


class _DensityNeighborhood:
    """Lazy exact displacement scaling; keep large source spectra read-only."""
    def __init__(self, original, scales):
        self.original = original
        self.scales = scales
        self.shape = original.shape
        if self.shape[0] != 1+6*len(scales):
            raise ValueError('density neighborhood inventory differs')

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.shape[0]:
            raise IndexError('density neighborhood requires a valid scalar point')
        scale = 1. if index == 0 else self.scales[(index-1)//6]
        if scale == 1.:
            return self.original[index]
        fiducial = self.original[0]
        return fiducial+scale*(self.original[index]-fiducial)


def rescale_density_source(source, factor):
    """Return a new physical source with density-dependent terms transformed.

    Arrays not changed by density may share read-only use of the original data.
    No existing source array or metadata is modified.
    """
    factor = _factor(factor)
    inputs = deepcopy(source['inputs'])
    meta = inputs['metadata']
    if PFS not in meta['active_populations'] or meta['nbar'].get(PFS, 0.) <= 0:
        raise ValueError('density change requires an observed PFS population')
    names = meta['local_names']
    if any(value != 0. for name, value in inputs['fiducials'].items()
           if name.split(':')[-1] in AUTO_STOCH | CROSS_STOCH):
        raise ValueError('exact density transform requires zero stochastic fiducials')
    scales = density_column_scales(names, factor)
    original = source['arrays']
    if (original['D'].shape[-1] != len(names)
            or not np.array_equal(original['fiducial'], [inputs['fiducials'][n] for n in names])):
        raise ValueError('density source parameter/fiducial layout differs')
    arrays = dict(original)
    if factor != 1.:
        meta['nbar'][PFS] *= factor
        for key in ('D', 'D_intrinsic', 'D_geometry'):
            if key in original:
                arrays[key] = original[key]*scales[None, None, :]
        for key in ('fd_plus', 'fd_minus'):
            arrays[key] = original[key].copy()
            for i in np.flatnonzero(scales != 1.):
                arrays[key][i] = original['mean']+scales[i]*(original[key][i]-original['mean'])
        arrays['fd_derivatives'] = ((arrays['fd_plus']-arrays['fd_minus'])
                                    /(2*arrays['fd_steps'][:, :, None, None]))
        _, arrays['C'] = covariance_from_spectra(arrays['pkmu'], meta,
                                                arrays['k'], arrays['dk'],
                                                n_mu=inputs['configuration']['n_mu'])
        if 'F_like' in arrays:
            arrays['F_like'] = fisher_from_shells(arrays['D'], arrays['C'], arrays['k'])
    return {**source, 'inputs':inputs, 'arrays':arrays,
            'neighborhood_spectra':_DensityNeighborhood(source['neighborhood_spectra'], scales),
            'density_transform':{'factor':factor, 'column_scales':scales.tolist(),
                                 'zero_stochastic_fiducials':True}}


def density_cell(source, profile, expected, grid, factor):
    """Rebuild and physically check one changed radial cell before inference."""
    varied = rescale_density_source(source, factor)
    support=varied['inputs']['analytic_zero_response']['support']
    existing={(n,tuple(pair),ell) for n,pair,ell in support}
    support.extend(row for row in gamma3_zero_support(varied['inputs']) if row not in existing)
    row = deepcopy(expected)
    if PFS not in row['active_populations']:
        raise ValueError('expected density cell has no PFS population')
    row['nbar'][PFS] *= _factor(factor)
    part = project_cell(varied, profile, row, grid)
    part['density_transform'] = varied['density_transform']
    return part
