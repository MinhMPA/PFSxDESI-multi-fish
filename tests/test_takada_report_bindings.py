"""The scientific report must authenticate the saved six-coordinate audits."""
import json
from pathlib import Path
import pytest


def audit_fixture(tmp_path):
    from pfsfog.elf_survey_evidence import file_digest
    source=Path(__file__).resolve().parents[1]/'results/takada-density-full-20261014-02/extended/loop2048'
    manifest=json.loads((source/'C/manifest.json').read_text())
    name=next(iter(manifest['physical_checks']))
    audit=json.loads((source/name).read_text())
    destination=tmp_path/name
    destination.write_text(json.dumps(audit))
    cell=next(c for c in manifest['likelihood']['cells'] if name=='C-'+c['metadata']['cell_id']+'-physical.json')
    return destination,audit,{'physical_checks':{name:file_digest(destination)},'likelihood':{'cells':[cell]}}


def test_report_binds_saved_extended_audit_and_rejects_changed_result(tmp_path):
    from scripts.report_takada_revision import verify_extended_audits
    path,audit,manifest=audit_fixture(tmp_path)
    bindings={}
    verify_extended_audits(tmp_path,'C',manifest,bindings)
    assert str(path) in bindings
    audit['derivative_check']['status']='failed'
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='audit changed'):
        verify_extended_audits(tmp_path,'C',manifest,{})


@pytest.mark.parametrize('failure',['missing_h','fixed_h','missing_audit'])
def test_report_rejects_incomplete_six_coordinate_audit(tmp_path,failure):
    from scripts.report_takada_revision import verify_extended_audits
    from pfsfog.elf_survey_evidence import file_digest
    path,audit,manifest=audit_fixture(tmp_path)
    if failure=='missing_h':
        audit['derivative_check']['column_results']=[r for r in audit['derivative_check']['column_results'] if r['column']!='h']
    elif failure=='fixed_h':
        audit['mapping']['fixed_cosmology']=['h']
    else:
        manifest['physical_checks']={}
    if failure!='missing_audit':
        path.write_text(json.dumps(audit));manifest['physical_checks'][path.name]=file_digest(path)
    with pytest.raises(ValueError,match='six-coordinate|inventory'):
        verify_extended_audits(tmp_path,'C',manifest,{})
