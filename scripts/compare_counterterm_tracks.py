"""Compare authenticated saved CT-A and CT-B forecasts without re-inference.

Only NumPy and the standard library are used. In particular, no model source
is imported from either physical campaign or from another worktree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

GRIDS = ('loop2048', 'loop4096', 'angular64', 'shell304', 'final-loop')
PUBLIC = {'A': 'A', 'B': 'B', 'D': 'C', 'C': 'D'}
ANCESTOR = 'b61780b6b9591b57123be39159e1f72753c60ea8'
HEADS = ('de66629e5b32c0021a31133715d18964dc68968e',
         '4f0a99b8cb8a8635a506e98f973119f5f3561295')
ACCEPTANCES = ('acceptance/d6f92e202c8e1ab31ab7d3fb18a45f59aceb6c682f36a63ecefab2be14272412',
               'verification/acceptance-1792542518796897000')
RECEIPTS = ('a41169d6ef0431444ced3fd10298c4fecd08916aeb554e08c1c6ee132111c60e',
            '34b9bd85c55f9ca15f48abe77c44524ec09693ce225df79f51250aa9cceaf0f4')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify_map(bindings, historical=None):
    for path, sha in bindings.items():
        if historical == (path, sha):
            continue
        require(digest(path) == sha, 'changed binding: '+str(path))
    return len(bindings)


def absolute_bindings(value):
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).startswith('/') and isinstance(item, str) and len(item) == 64:
                require(all(c in '0123456789abcdef' for c in item), 'invalid digest')
                found[key] = item
            else:
                for path, sha in absolute_bindings(item).items():
                    require(path not in found or found[path] == sha, 'conflicting binding')
                    found[path] = sha
    return found


def compare_array(left, right, label, inferred=False):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape, 'shape differs: '+label)
    require(np.all(np.isfinite(left)) and np.all(np.isfinite(right)), 'nonfinite: '+label)
    rtol, atol = (1e-8 if inferred else 1e-10), 1e-10
    delta = np.abs(left-right)
    record = {'max_abs': float(delta.max(initial=0)), 'rtol': rtol, 'atol': atol,
              'relative_norm': float(np.linalg.norm(delta)/max(np.linalg.norm(left), 1e-300))}
    require(np.allclose(left, right, rtol=rtol, atol=atol), 'array differs: '+label+' '+str(record))
    return record


def common_indices(left, right):
    require(len(left) == len(set(left)) and len(right) == len(set(right)), 'duplicate parameter')
    require(set(left) <= set(right), 'missing common parameter')
    require(all(n.endswith(':c_fog_extra') for n in set(right)-set(left)), 'unexpected parameter')
    return [right.index(n) for n in left]


def compare_engine(left, right):
    left, right = dict(left), dict(right)
    labels = [x.pop('counterterm_prescription', None) for x in (left, right)]
    require(all(x in (None, 'CT-A-shared-field-leading-v1') for x in labels), 'unknown engine track label')
    require(left == right, 'different physical engine settings')
    return labels


def require_coverage(actual, expected, label):
    require(len(actual) == len(set(actual)) and set(actual) == set(expected), 'incomplete/duplicate '+label)


def error_gains(rows):
    indexed = {r['public_case']: r for r in rows}
    require_coverage([r['public_case'] for r in rows], PUBLIC.values(), 'gain cases')
    return {kind: {x+'_to_'+y: {q: 100*(1-indexed[y]['sigma_'+kind][q]/den)
                              for q, den in indexed[x]['sigma_'+kind].items()}
                   for x, y in (('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'D'))}
            for kind in ('marginalized', 'nuisance_fixed')}


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def authenticate(output, index):
    root = output.parents[1]
    request = read(output/'request.json'); parent = digest(output/'request.json')
    state = read(output/'state.json'); comparison = read(output/'comparison.json')
    require(request['track'] == ('CT-A', 'CT-B')[index], 'wrong physical track')
    require(state['status'] == comparison['status'] == 'qualified' and not state['active_jobs'], 'unfinished physical run')
    require(state['request_sha256'] == comparison['request_sha256'] == parent, 'changed physical request')
    require(state['comparison_sha256'] == digest(output/'comparison.json'), 'changed comparison')
    acceptance = output/ACCEPTANCES[index]
    require(digest(acceptance/'receipt.json') == RECEIPTS[index], 'wrong acceptance receipt')
    receipt = read(acceptance/'receipt.json'); aq = read(acceptance/'request.json')
    require(receipt['status'] == 'qualified' and receipt['request_sha256'] == digest(acceptance/'request.json'), 'unqualified acceptance')
    require(receipt.get('physical_request_sha256', receipt.get('parent_request_sha256')) == parent, 'wrong acceptance parent')
    require(digest(acceptance/'comparison.json') == receipt['comparison_sha256'] == digest(output/'comparison.json'), 'acceptance comparison differs')
    runner = aq['historical_runner']; runner_sha = runner.get('sha256', runner.get('original_sha256'))
    runner_path = str(root/'scripts/run_counterterm_comparison.py')
    require(runner['path'] == runner_path and runner['commit'] == request['source_head'], 'wrong runner exception')
    snapshot = Path(runner['snapshot'])
    if not snapshot.is_absolute():
        snapshot = acceptance/snapshot
    blob = subprocess.check_output(['git', 'show', runner['commit']+':scripts/run_counterterm_comparison.py'], cwd=root)
    require(hashlib.sha256(blob).hexdigest() == digest(snapshot) == runner_sha == request['bindings'][runner_path], 'wrong historical runner')
    bindings = absolute_bindings({'bindings': request['bindings'], 'sources': request['sources'], 'historical_source': request['historical_source']})
    require(bindings.pop(runner_path) == runner_sha, 'runner binding absent')
    physical_count = verify_map(bindings)
    # CT-B retains the original runner in `bindings` and its revised acceptance
    # bytes in `acceptance_sources`. Authenticate each inventory separately;
    # their one authorized difference must never weaken another source check.
    acceptance_counts = {name: verify_map(absolute_bindings(aq.get(name, {})), historical=(runner_path, runner_sha))
                         for name in ('input_files', 'checker_files', 'bindings', 'acceptance_sources', 'preserved_artifacts')
                         if name in aq}
    require(git(root, 'merge-base', '--is-ancestor', ANCESTOR, HEADS[index]) == '', 'wrong base ancestry')
    require(git(root, 'merge-base', '--is-ancestor', request['source_head'], HEADS[index]) == '', 'wrong physical ancestry')
    git(root, 'merge-base', '--is-ancestor', HEADS[index], 'HEAD')
    # Descendant commits may add the comparison/report but may not alter any
    # source bound by the physical or append-only acceptance requests.
    return request, comparison, {'worktree': str(root), 'head': git(root, 'rev-parse', 'HEAD'),
            'reviewed_head': HEADS[index], 'physical_source_head': request['source_head'],
            'physical_request_sha256': parent, 'acceptance_receipt_sha256': RECEIPTS[index],
            'acceptance_request_sha256': digest(acceptance/'request.json'),
            'physical_bindings_checked': physical_count, 'acceptance_bindings_checked': acceptance_counts,
            'historical_runner': {k: v for k, v in runner.items()}, 'active_jobs': state['active_jobs']}


def physical_compare(a, b, parent_a, parent_b):
    records = []
    axes = {'D': -1, 'D_intrinsic': -1, 'D_geometry': -1, 'fiducial': 0,
            'fd_plus': 0, 'fd_minus': 0, 'fd_steps': 0, 'fd_derivatives': 0}
    for grid in GRIDS:
        aa = {p.parent.name: p for p in (a/'inference'/grid/'physical').glob('*/receipt.json')}
        bb = {p.parent.name: p for p in (b/'inference'/grid/'physical').glob('*/receipt.json')}
        require(len(aa) == len(bb) == 40 and set(aa) == set(bb), 'wrong physical cell coverage '+grid)
        for key in sorted(aa):
            pa, pb = aa[key].parent, bb[key].parent
            for p, parent in ((pa, parent_a), (pb, parent_b)):
                receipt = read(p/'receipt.json')
                require(receipt['status'] == 'passed' and receipt['request_sha256'] == parent, 'unqualified physical cell')
                verify_map({str(p/n): sha for n, sha in receipt['files'].items()})
                verify_map(absolute_bindings(receipt['source_files']))
                verify_map(absolute_bindings(receipt['projection_files']))
            ia, ib = read(pa/'inputs.json'), read(pb/'inputs.json')
            names = ia['metadata']['local_names']; other = ib['metadata']['local_names']; indices = common_indices(names, other)
            for field in ('configuration', 'units', 'h_ref'):
                require(ia[field] == ib[field], 'different physical input '+field)
            engine_labels = compare_engine(ia['engine_settings'], ib['engine_settings'])
            ma, mb = dict(ia['metadata']), dict(ib['metadata']); ma.pop('local_names'); mb.pop('local_names')
            require(ma == mb, 'different cell geometry/observations')
            require(all(ia['fiducials'][n] == ib['fiducials'][n] for n in names), 'different fiducials')
            require(all(ib['fiducials'][n] == 0 for n in set(other)-set(names)), 'nonzero U')
            checks = {}
            with np.load(pa/'arrays.npz', allow_pickle=False) as za, np.load(pb/'arrays.npz', allow_pickle=False) as zb:
                require(set(za.files) == set(zb.files), 'different physical arrays')
                for name in za.files:
                    av, bv = za[name], zb[name]
                    if name in axes:
                        bv = np.take(bv, indices, axis=axes[name])
                    checks[name] = compare_array(av, bv, grid+'/'+key+'/'+name)
            records.append({'grid': grid, 'physical_key': key, 'cell_id': ma['cell_id'], 'descriptive_engine_labels': engine_labels,
                            'inputs': {str(pa/'receipt.json'): digest(pa/'receipt.json'), str(pb/'receipt.json'): digest(pb/'receipt.json')},
                            'checks': checks})
    return records


def inference_compare(a, b):
    records = []
    for grid in GRIDS:
        for internal, public in PUBLIC.items():
            pa = a/'inference'/grid/'profiles'/internal
            pb = b/'inference'/grid/'treatments/fixed_zero/profiles'/internal
            ma, mb = read(pa/'manifest.json'), read(pb/'manifest.json')
            require(ma['likelihood']['names'] == mb['likelihood']['names'], 'different fixed-zero parameter names')
            require(ma['parameters'] == mb['parameters'], 'different common priors')
            for field in ('configuration', 'reference_cosmology', 'parameters'):
                require(ma['likelihood'][field] == mb['likelihood'][field], 'different likelihood '+field)
            ca, cb = ma['likelihood']['cells'], mb['likelihood']['cells']
            require(len(ca) == len(cb), 'different survey cells')
            require_coverage([c['metadata']['cell_id'] for c in ca], [c['metadata']['cell_id'] for c in cb], 'survey cells')
            for x, y in zip(ca, cb):
                require(x['metadata'] == y['metadata'] and x['names'] == y['names'], 'different observations/common responses')
            checks = {}
            with np.load(pa/'arrays.npz', allow_pickle=False) as za, np.load(pb/'arrays.npz', allow_pickle=False) as zb:
                require(set(za.files) == set(zb.files), 'different inference objects')
                for name in za.files:
                    checks[name] = compare_array(za[name], zb[name], grid+'/'+public+'/'+name,
                        inferred=name.startswith('result/') and ('cov' in name))
            records.append({'grid': grid, 'public_case': public, 'cells': len(ca), 'checks': checks,
                            'inputs': {str(p/n): digest(p/n) for p in (pa, pb) for n in ('manifest.json', 'arrays.npz', 'constraints.json')}})
    return records


def historical_regression(a):
    old = a.parent/'counterterm-leading-20261020-03'; files = {}
    request = old/'request.json'; state = read(old/'state.json')
    require(digest(request) == state['request_sha256'], 'changed retained request03')
    for profile in ('B', 'D'):
        record = old/'inference/loop2048/parts'/profile/'bin01-radial00-overlap/record.json'
        physical = Path(read(record)['physical_receipt'])
        receipt = read(physical)
        require(receipt['request_sha256'] == digest(request), 'wrong retained request03 physical parent')
        bound = {str(physical.parent/n): sha for n, sha in receipt['files'].items()}
        verify_map(bound); verify_map(absolute_bindings(receipt['projection_files']))
        files.update(bound); files[str(physical)] = digest(physical); files[str(record)] = digest(record)
    files[str(request)] = digest(request)
    return {'status': 'retained failed/superseded request; regression input only', 'files': files,
            'test': 'tests/test_counterterm_shared_observation.py::test_saved_two_tracer_cell_keeps_signed_differences_and_passes_qualification'}


def compare(a, b, protected):
    qa, ra, auth_a = authenticate(a, 0); qb, rb, auth_b = authenticate(b, 1)
    for field in ('source_root', 'old_study', 'historical_request_sha256', 'grids', 'public_labels', 'numerical_contract'):
        require(qa[field] == qb[field], 'different campaign input '+field)
    audit = read(protected); verify_map(audit)
    physical = physical_compare(a, b, digest(a/'request.json'), digest(b/'request.json'))
    inference = inference_compare(a, b)
    variants = {'old': [r for r in ra['rows'] if r['model'] == 'old'],
                'CT-A': [r for r in ra['rows'] if r['model'] == 'CT-A']}
    for treatment in ('fixed_zero', 'sigma_100', 'sigma_400', 'sigma_1600'):
        variants['CT-B/'+treatment] = [r for r in rb['rows'] if r['model'] == 'CT-B' and r['extra_fog'] == treatment]
    for name, rows in variants.items():
        require_coverage([(r['grid'], r['public_case']) for r in rows],
                         {(g, p) for g in GRIDS for p in PUBLIC.values()}, name)
    gains = {name: {g: error_gains([r for r in rows if r['grid'] == g]) for g in GRIDS} for name, rows in variants.items()}
    for x in [r for r in variants['CT-A']]:
        y = next(y for y in variants['CT-B/fixed_zero'] if y['grid'] == x['grid'] and y['public_case'] == x['public_case'])
        for kind in ('marginalized', 'nuisance_fixed'):
            for q in x['sigma_'+kind]:
                compare_array(x['sigma_'+kind][q], y['sigma_'+kind][q], 'cosmological uncertainty', inferred=True)
    return {'status': 'passed', 'checker_sha256': digest(__file__), 'authentication': [auth_a, auth_b],
            'protected': {'before_file': str(protected), 'before_sha256': digest(protected), 'checked': len(audit), 'changed': [], 'verified_files': audit},
            'physical_cells': physical, 'survey_results': inference, 'variants': variants, 'gains': gains,
            'historical_regression': historical_regression(a)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ct-a', type=Path, required=True)
    parser.add_argument('--ct-b', type=Path, required=True)
    parser.add_argument('--protected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'comparison output exists; choose a new append-only path')
    result = compare(args.ct_a.resolve(), args.ct_b.resolve(), args.protected.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(json.dumps({'status': result['status'], 'physical_cells': len(result['physical_cells']),
                      'survey_results': len(result['survey_results']), 'protected': result['protected']['checked'],
                      'output_sha256': digest(args.output)}))


if __name__ == '__main__':
    main()
