"""The complete physical support, with shared nominal-bin populations."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("case,total_points", [("B", 3200), ("C", 3752)])
def test_exact_survey_support_and_signed_neighborhood_counts(case, total_points):
    from pfsfog.elf_survey import survey_cases
    from pfsfog.revision_forecast import RevisionConfig, make_shell_grid
    from pfsfog.revision_validation_physical import expected_support_schedule
    from pfsfog.surveys import default_survey_group
    from scripts.validate_elf_engine import elf_cosmology_spec
    from pfsfog.cosmo import FIDUCIAL
    config = RevisionConfig(case, cosmology_basis="extended6", dk=.00125)
    cosmo = SimpleNamespace(params=FIDUCIAL, sigma8=lambda z: .6)
    cases = survey_cases(config, cosmo)
    expected = expected_support_schedule(config, default_survey_group(),
                                         space=elf_cosmology_spec(FIDUCIAL))
    assert len(cases) == len({item["cell_id"] for item in cases}) == 20
    assert len(make_shell_grid(.01, .20, config.dk).centers) == 152
    assert [item["expected"] for item in cases] == expected
    assert sum(1+6*len(row["local_names"]) for row in expected) == total_points
    assert {item["region"] for item in cases} == {"overlap", "nonoverlap"}
    for item in cases:
        row = item["expected"]
        assert item["region"] == row["region"]
        assert item["cell_id"] == row["cell_id"]
        if row["radial_bin"][0] >= 1.6 or case == "B" or item["region"] == "nonoverlap":
            assert "PFS-ELG" not in row["active_populations"]
        assert all(fid.c_tilde > 0 for fid in item["fiducials"].values())
    for ibin in range(8):
        selected = [item for item in cases if item["expected"]["nominal_bin_index"] == ibin]
        assert all(item["fiducials"] == selected[0]["fiducials"] for item in selected)
    if case == "C":
        dense = next(item for item in cases if item["cell_id"] == "bin03-radial00-overlap")
        assert len(dense["expected"]["active_populations"]) == 4
        assert len(dense["expected"]["retained_pairs"]) == 10


def test_nonoverlap_measurement_preserves_actual_identity_and_volume(tmp_path):
    from pfsfog.elf_survey import survey_cases
    from pfsfog.revision_forecast import RevisionConfig
    from pfsfog.cosmo import FiducialCosmology
    from pfsfog.one_loop import make_one_loop_engine
    from scripts.validate_elf_engine import measure_cell
    config = RevisionConfig("B", cosmology_basis="extended6", kmax=.03, dk=.01)
    cosmo = FiducialCosmology()
    case = survey_cases(config, cosmo)[1]
    assert case["region"] == "nonoverlap"
    cell, result = measure_cell(case, config, cosmo, make_one_loop_engine(), tmp_path)
    assert cell.metadata["cell_id"] == "bin00-radial00-nonoverlap"
    assert cell.metadata["region"] == "nonoverlap"
    assert cell.metadata["volume"] == case["cell"].nonoverlap_volume
    assert cell.metadata["volume"] != case["cell"].overlap_volume
    assert result["status"] == "passed"


def test_other_scopes_are_not_silently_promoted():
    from pfsfog.elf_survey import survey_cases
    from pfsfog.revision_forecast import RevisionConfig
    with pytest.raises(ValueError, match="B/C.*extended6"):
        survey_cases(RevisionConfig("C"), None)
