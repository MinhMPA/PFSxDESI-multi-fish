"""Baseline survey cells for the ELF six-coordinate physical assessment."""
from .cosmology_space import CosmologySpec, make_cosmology_spec
from .eft_params import tracer_fiducials
from .revision_validation_physical import expected_support_schedule
from .survey_geometry import ShellGeometry, support_cells
from .surveys import default_survey_group


def elf_cosmology_spec(reference):
    """Accepted six-coordinate steps, independent of old AP-table labels."""
    base = make_cosmology_spec(reference, extended=True)
    steps = {**base.fd_steps, "h": (.000125, .0000625, .00003125)}
    return CosmologySpec(base.names, base.h_ref, steps, base.unit_convention)


def survey_cases(config, cosmo):
    """Keep the original nominal-bin populations across all support regions."""
    if config.case not in ("B", "C") or config.cosmology_basis != "extended6":
        raise ValueError("this physical stage requires B/C in extended6 coordinates")
    group = default_survey_group()
    space = make_cosmology_spec(cosmo.params, extended=True)
    expected = expected_support_schedule(config, group, space=space)
    expected_by_id = {row["cell_id"]: row for row in expected}
    geometry = ShellGeometry.from_group(group)
    output = []
    for ibin, nominal in enumerate(config.nominal_bins):
        baseline = support_cells(group, nominal, include_pfs=True,
                                 exclude_elg1=False, geometry=geometry)
        population = sorted({name for cell in baseline for name in cell.overlap_tracers})
        z = sum(nominal)/2
        sigma8 = float(cosmo.sigma8(z))
        b_ref = group.desi_tracers["DESI-ELG"].b1_of_z(z) if "DESI-ELG" in population else 1.3
        fiducials = {name: tracer_fiducials(name, group.all_surveys[name].b1_of_z(z),
                        sigma8, b1_ref=b_ref, r_sigma_v=config.r_sigma_v)
                     for name in population}
        selected = support_cells(group, nominal, include_pfs=True,
                                 exclude_elg1=config.exclude_elg1, geometry=geometry)
        for icell, cell in enumerate(selected):
            for region in ("overlap", "nonoverlap"):
                identity = f"bin{ibin:02d}-radial{icell:02d}-{region}"
                output.append({"label": f"{config.case}:{identity}", "cell_id": identity,
                    "region": region, "scope": "baseline_survey_physical",
                    "cell": cell, "fiducials": fiducials,
                    "expected": expected_by_id[identity]})
    if [item["cell_id"] for item in output] != [row["cell_id"] for row in expected]:
        raise ValueError("survey enumeration differs from independent support schedule")
    return output
