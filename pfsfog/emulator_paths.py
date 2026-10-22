"""Explicit, byte-checked emulator inputs and the four approved relocations.

Only standard-library imports: historical verification can load this file
without importing the numerical package or changing any historical metadata.
"""
from pathlib import Path
import hashlib
from importlib.machinery import PathFinder
import importlib.metadata
import json
import sys

PACKAGE_ROOT=Path('/Users/nguyenmn/cosmopower-jax')
MODEL_ROOT=PACKAGE_ROOT/'cosmology/jense2024'
MODEL='jense_2023_camb_mnu'
HISTORICAL_ROOT=Path('/Users/nguyenmn/cosmopower-jax-for-pfs')
MODEL_FILES={
    'jense_2023_pk_camb_mnu.yaml':'771bd7184c430540f167bcb39e8e1bdeb4725fa158aad97d5cbfb3d5ed915e3e',
    'jense_2023_camb_mnu/networks/jense_2023_camb_mnu_Pk_lin.npz':'a8b5b12d11131fce8c64c9d1e0f65b6dd0a18f95541943a0f2f2e04738fad3e2',
    'jense_2023_camb_mnu/networks/jense_2023_camb_mnu_sigma8.npz':'6b2d8039497a11ecd6e36237aaedcc54cddb02e554c2321b6c1d1feefdd35b95'}
_EMPTY='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
_PACKAGE_FILES={'__init__.py':_EMPTY,'pca_utils/__init__.py':_EMPTY,'trained_models/__init__.py':_EMPTY,
                'cosmopower_jax.py':'89790e4de7c9078244f586b54d83f866ab962da508f4060c712ee74c3d6199c1'}
_RELOCATION_PIN='1dfcc93f8df4085a00374d8a0fdd52bff92cc5a6c3162b3c30a48b498f800876'


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_bytes(path,expected):
    if not Path(path).is_file(): raise ValueError('missing or unlisted historical/input file: '+str(path))
    if _sha(path)!=expected: raise ValueError('input bytes differ: '+str(path))


def historical_emulator_inputs():
    return {str(HISTORICAL_ROOT/'cosmology/jense2024'/p):h for p,h in MODEL_FILES.items()}


def active_emulator_inputs():
    result={str(MODEL_ROOT/p):h for p,h in MODEL_FILES.items()}
    for p,h in result.items(): _verify_bytes(p,h)
    return result


def active_emulator_provenance():
    package=PACKAGE_ROOT/'cosmopower_jax'
    files={str(package/p):h for p,h in _PACKAGE_FILES.items()}
    if {str(p) for p in package.rglob('*.py')}!=set(files):
        raise ValueError('active emulator package source inventory differs')
    for p,h in files.items(): _verify_bytes(p,h)
    specs={}
    origins={}
    for name,path in (('cosmopower_jax',package/'__init__.py'),
                      ('cosmopower_jax.cosmopower_jax',package/'cosmopower_jax.py')):
        search=None if '.' not in name else specs['cosmopower_jax'].submodule_search_locations
        spec=PathFinder.find_spec(name,search)
        if spec is None or spec.origin is None or Path(spec.origin).resolve()!=path:
            raise ValueError('active emulator package resolution origin differs: '+name)
        loaded=sys.modules.get(name)
        if loaded is not None:
            loaded_spec=getattr(loaded,'__spec__',None)
            if (getattr(loaded,'__file__',None)!=str(path) or loaded_spec is None
                    or loaded_spec.origin!=str(path)):
                raise ValueError('active emulator loaded package origin differs: '+name)
            if name=='cosmopower_jax' and list(loaded.__path__)!=[str(package)]:
                raise ValueError('active emulator package search origin differs')
        specs[name]=spec;origins[name]={'origin':str(path),'sha256':files[str(path)]}
    version=importlib.metadata.version('cosmopower-jax')
    if version!='0.5.7': raise ValueError('active emulator package version differs')
    return {'schema_version':'pfs-desi-active-emulator-v1','model_root':str(MODEL_ROOT),
            'model':MODEL,'package_version':version,'package_files':files,'package_origins':origins}


def approved_relocation_record():
    path=Path(__file__).resolve().parents[1]/'docs/revision/evidence/extended-cosmology-study/ap-global-continuation-2026-10-05/RELOCATION_AUTHENTICATION.json'
    _verify_bytes(path,_RELOCATION_PIN)
    return json.loads(path.read_text())


def verify_historical_relocation(recorded,relocation_record):
    """Verify original keys without rewriting them or granting prefix exceptions."""
    expected=[{'historical_path':str(HISTORICAL_ROOT/'cosmology/jense2024'/p),
               'active_path':str(MODEL_ROOT/p),'sha256':h} for p,h in MODEL_FILES.items()]
    expected.append({'historical_path':str(HISTORICAL_ROOT/'cosmopower_jax/__init__.py'),
                     'active_path':str(PACKAGE_ROOT/'cosmopower_jax/__init__.py'),'sha256':_EMPTY})
    rows=relocation_record.get('relocated_paths',[])
    if not isinstance(rows,list) or len(rows)!=4 or any(row not in expected for row in rows) or len({row['historical_path'] for row in rows})!=4:
        raise ValueError('unapproved historical relocation mapping or expected hash')
    active=active_emulator_provenance()
    if (relocation_record.get('active_package_python_files')!=active['package_files']
            or relocation_record.get('package_origins',{}).get('cosmopower_jax')!=active['package_origins']['cosmopower_jax']
            or relocation_record.get('package_versions',{}).get('cosmopower-jax')!=active['package_version']):
        raise ValueError('relocation package origin/source/version differs')
    allowed={row['historical_path']:row for row in expected};used=[]
    for path,sha in recorded.items():
        if path in allowed:
            row=allowed[path]
            if row['sha256']!=sha: raise ValueError('historical expected hash differs')
            _verify_bytes(row['active_path'],sha);used.append(row.copy())
        else: _verify_bytes(path,sha)
    return {'schema_version':'pfs-desi-historical-relocation-verification-v1',
            'historical_exact_path_replay':False,'verified_file_count':len(recorded),
            'recorded_hashes_sha256':hashlib.sha256(json.dumps(recorded,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'relocated_paths':used,'active_package':active}
