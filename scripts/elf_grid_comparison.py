"""Saved-array grid comparisons; never change a measured likelihood or prior."""
from copy import deepcopy
import hashlib
import json

import numpy as np

LOCAL_TOLERANCE=5.e-4
GLOBAL_TOLERANCE=.01
ZERO_FRACTION=1.e-12
TREATMENTS=('conditional3','release_h','release_omega_b','release_n_s',
            'marginal6','marginal6_weak10','marginal6_no_extra')
COSMOLOGY=['ln10_10_As','Mnu','Omegam','h','omega_b','n_s']
KINDS=('marginalized','nuisance_fixed')


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _grid(k,dk):
    k=np.asarray(k,dtype=float);dk=np.broadcast_to(np.asarray(dk,dtype=float),k.shape)
    if (k.ndim!=1 or len(k)<2 or not np.isfinite(k).all() or not np.isfinite(dk).all()
            or np.any(np.diff(k)<=0) or np.any(dk<=0)):
        raise ValueError('invalid ordered k grid or shell bandwidth')
    return k,dk


def _arrays(k,mean,D,C):
    mean,D,C=[np.asarray(x,dtype=float) for x in (mean,D,C)]
    if (mean.ndim!=2 or D.ndim!=3 or C.ndim!=3 or mean.shape[0]!=len(k)
            or D.shape[:2]!=mean.shape or C.shape!=(len(k),mean.shape[1],mean.shape[1])
            or not all(np.isfinite(x).all() for x in (mean,D,C))):
        raise ValueError('invalid or nonfinite local array shape')
    if not np.allclose(C,C.swapaxes(-1,-2),rtol=1e-12,atol=0):
        raise ValueError('asymmetric covariance')
    try:np.linalg.cholesky(C)
    except np.linalg.LinAlgError as error:raise ValueError('nonpositive covariance') from error
    return mean,D,C


def align_shells(k_coarse,dk_coarse,k_fine,dk_fine,mean,derivatives,covariance):
    """Interpolate continuous functions; restore coarse covariance bandwidth."""
    kc,dc=_grid(k_coarse,dk_coarse);kf,df=_grid(k_fine,dk_fine)
    mean,D,C=_arrays(kf,mean,derivatives,covariance)
    if kc[0]<kf[0] or kc[-1]>kf[-1]:
        raise ValueError('shell comparison would extrapolate')
    def interpolate(a):
        flat=a.reshape(len(kf),-1)
        return np.column_stack([np.interp(kc,kf,col) for col in flat.T]).reshape((len(kc),)+a.shape[1:])
    return interpolate(mean),interpolate(D),interpolate(C*df[:,None,None])/dc[:,None,None]


def compare_local(coarse,fine,*,shell=False):
    kc,dc=_grid(coarse['k'],coarse['dk']);kf,df=_grid(fine['k'],fine['dk'])
    m,D,C=_arrays(kc,coarse['mean'],coarse['D'],coarse['C'])
    mf,Df,Cf=_arrays(kf,fine['mean'],fine['D'],fine['C'])
    names=coarse['names']
    if (names!=fine['names'] or len(names)!=D.shape[-1] or len(set(names))!=len(names)
            or Df.shape[1:]!=D.shape[1:]):
        raise ValueError('local parameter or observation layout changed')
    if shell:mf,Df,Cf=align_shells(kc,dc,kf,df,mf,Df,Cf)
    elif not np.array_equal(kc,kf) or not np.array_equal(dc,df):
        raise ValueError('direct comparison requires identical k and dk')
    chol=np.linalg.cholesky(C)
    white=lambda a:np.linalg.solve(chol,a)
    mean_scale=float(np.linalg.norm(white(m[...,None])))
    if not np.isfinite(mean_scale) or mean_scale<=0:raise ValueError('nonpositive mean norm')
    norm=np.linalg.norm(white(D),axis=(0,1))
    fnorm=np.linalg.norm(white(Df),axis=(0,1))
    delta=np.linalg.norm(white(Df-D),axis=(0,1))
    columns={}
    for name,den,fden,num in zip(names,norm,fnorm,delta):
        both_zero=max(den,fden)<=ZERO_FRACTION*mean_scale
        normalization='full_mean_for_both_zero' if both_zero else 'coarse_response'
        scale=mean_scale if both_zero else den
        value=float(num/scale) if scale>0 else None
        columns[name]={'change':value,'normalization':normalization,
            'coarse_norm':float(den),'fine_norm':float(fden),'difference_norm':float(num),
            'status':'passed' if value is not None and value<LOCAL_TOLERANCE else 'failed'}
    left=white(Cf-C)
    wc=white(left.swapaxes(-1,-2)).swapaxes(-1,-2)
    covariance_change=float(np.max(np.linalg.norm(wc,ord=2,axis=(-2,-1))))
    mean_change=float(np.linalg.norm(white((mf-m)[...,None]))/mean_scale)
    passed=(mean_change<LOCAL_TOLERANCE and covariance_change<LOCAL_TOLERANCE
            and all(v['status']=='passed' for v in columns.values()))
    return {'status':'passed' if passed else 'failed','tolerance':LOCAL_TOLERANCE,
        'alignment':'linear_continuous_kernel_to_coarse' if shell else 'identical_k',
        'mean_relative':mean_change,'mean_norm':mean_scale,
        'covariance_maximum_spectral_change':covariance_change,'columns':columns,
        'full_forecast_qualification':'pending'}


def _free(t):
    if t.startswith('marginal6'):return COSMOLOGY
    return COSMOLOGY[:3]+([t.removeprefix('release_')] if t.startswith('release_') else [])


def _field_values(row,free):
    if set(row)!= {'cosmology','growth'} or set(row['cosmology'])!=set(free) or len(row['growth'])!=8:
        raise ValueError('wrong cosmology labels or growth count')
    values={**{'cosmology/'+n:row['cosmology'][n] for n in free},
            **{'growth/'+str(i):v for i,v in enumerate(row['growth'])}}
    if any(not np.isfinite(v) or v<=0 for v in values.values()):
        raise ValueError('errors and ratios must be finite and positive')
    return values


def _constraints(summary):
    out=summary['outcomes'];ratios=summary['ratios'];values={};unqualified=[];statuses={}
    if set(out)!= {'B','C'} or set(ratios)!=set(TREATMENTS):raise ValueError('incomplete B/C outcomes or ratios')
    for case in ('B','C'):
        if set(out[case])!=set(TREATMENTS):raise ValueError('incomplete treatment set')
        for t,row in out[case].items():
            label=case+'/'+t;statuses[label]=row['status']
            if row['status']=='failed':
                if t in ('conditional3','marginal6'):raise ValueError('required control failed')
                if not row.get('error'):raise ValueError('failed treatment lacks explanation')
                unqualified.append(label);continue
            if row['status']!='passed' or row['free_cosmology']!=_free(t):raise ValueError('wrong treatment status or basis')
            for kind in KINDS:
                for field,value in _field_values(row[kind],_free(t)).items():values[label+'/'+kind+'/'+field]=value
    for t,row in ratios.items():
        label='ratios/'+t;failed=[c for c in ('B','C') if out[c][t]['status']=='failed']
        if failed:
            if row!={'status':'unavailable','failed_cases':failed}:raise ValueError('failed treatment ratio fabricated')
            unqualified.append(label);continue
        if row['status']!='available' or row['free_cosmology']!=_free(t):raise ValueError('ratio basis differs')
        for kind in KINDS:
            b=_field_values(out['B'][t][kind],_free(t));c=_field_values(out['C'][t][kind],_free(t))
            for direction,left,right in [('B_over_C',b,c),('C_over_B',c,b)]:
                for field,value in _field_values(row[kind][direction],_free(t)).items():
                    if value!=left[field]/right[field]:raise ValueError('saved ratio differs from constraints')
                    values[label+'/'+kind+'/'+direction+'/'+field]=value
    return values,unqualified,statuses


def compare_constraints(coarse,fine):
    c,cu,cs=_constraints(coarse);f,fu,fs=_constraints(fine)
    changed=[n for n in cs if cs[n]!=fs[n]]
    metrics=[]
    for label in sorted(set(c)&set(f)):
        delta=float(abs((f[label]-c[label])/c[label]))
        # An unresolved boundary is not evidence of a strict inequality. Cover
        # binary64 input rounding, subtraction and division conservatively.
        roundoff=8*np.finfo(float).eps*max(1.,abs(f[label]/c[label]))
        metrics.append({'label':label,'coarse':c[label],'fine':f[label],'relative_change':delta,
                        'roundoff_guard':float(roundoff),
                        'status':'passed' if delta+roundoff<GLOBAL_TOLERANCE else 'failed'})
    passed=not changed and all(m['status']=='passed' for m in metrics)
    worst=max(metrics,key=lambda m:m['relative_change'])
    return {'status':'passed' if passed else 'failed','tolerance':GLOBAL_TOLERANCE,
        'metrics':metrics,'maximum_relative_change':worst['relative_change'],'worst_label':worst['label'],
        'unqualified':sorted(set(cu)|set(fu)),'changed_treatment_status':changed,
        'full_forecast_qualification':'pending'}


def validate_pair_contract(coarse_request,fine_request,axis):
    """Compare all fixed request content, removing only checked grid differences."""
    if axis not in ('loop','angular','shell','combined'):raise ValueError('unknown refinement axis')
    copies=[deepcopy(coarse_request),deepcopy(fine_request)]
    c,f=[r['settings'] for r in copies]
    for r in copies:
        s=r['settings'];source=r['source'];d=source['details']
        if (set(s)!= {'nfft','n_mu','dk'} or any(not np.isfinite(v) or v<=0 for v in s.values())
                or any(isinstance(s[k],bool) or int(s[k])!=s[k] for k in ('nfft','n_mu'))
                or source['identity_hash']!=hashlib.sha256(_canonical(d)).hexdigest()
                or d['settings']!=s or any(d[key]['fft_n']!=s['nfft'] for key in ('model_flags','engine_settings'))
                or set(r['configurations'])!= {'B','C'}
                or any(config['dk']!=s['dk'] or config['n_mu']!=s['n_mu'] for config in r['configurations'].values())):
            raise ValueError('invalid or inconsistent source/grid records')
    changes={k for k in c if c[k]!=f[k]}
    allowed={'loop':{'nfft'},'angular':{'n_mu'},'shell':{'dk'},'combined':{'nfft','n_mu','dk'}}[axis]
    if (not changes or (axis=='combined' and len(changes)<2) or (axis!='combined' and changes!=allowed)
            or any(f[k] != c[k]*(.5 if k=='dk' else 2) for k in changes)):
        raise ValueError('wrong or undeclared grid refinement')
    original=[deepcopy(c),deepcopy(f)]
    for r in copies:
        d=r['source']['details'];r['source'].pop('identity_hash')
        for key in changes:
            r['settings'][key]=original[0][key];d['settings'][key]=original[0][key]
            if key=='nfft':
                d['model_flags']['fft_n']=original[0][key];d['engine_settings']['fft_n']=original[0][key]
            else:
                for config in r['configurations'].values():config[key]=original[0][key]
    if _canonical(copies[0])!=_canonical(copies[1]):raise ValueError('fixed physical content changed')
    return {'status':'passed','axis':axis,'changed_settings':sorted(changes),'coarse':original[0],'fine':original[1]}
