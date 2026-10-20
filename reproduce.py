"""Prepare the research files and verify saved numerical evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def extract_safe(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            path = Path(item.filename)
            target = (destination / path).resolve()
            if path.is_absolute() or ':' in item.filename or '\\' in item.filename or not target.is_relative_to(destination):
                raise ValueError('Unsafe archive member: ' + item.filename)
            if not item.is_dir() and target.exists() and target.read_bytes() != zipped.read(item):
                raise FileExistsError('Preparation would overwrite a local modification: ' + item.filename)
        if zipped.testzip() is not None:
            raise ValueError('Archive checksum failure')
        zipped.extractall(destination)


def assert_close(actual, expected, path='result', atol=1e-7, rtol=2e-10):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise AssertionError(path + ': value or type differs')
    elif isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(path + ': numeric mismatch')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(path + ': keys differ')
        for key in expected:
            assert_close(actual[key], expected[key], path + '.' + key, atol, rtol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(path + ': lengths differ')
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_close(a, b, path + '[' + str(i) + ']', atol, rtol)
    elif actual != expected:
        raise AssertionError(path + ': unsupported differing value')


def prepare(root=ROOT):
    root = Path(root)
    manifest = read(root / 'release_manifest.json')
    code = manifest['code_archive']
    if digest(root / code['name']) != code['sha256']:
        raise ValueError('Code archive differs from its manifest')
    extract_safe(root / code['name'], root)
    merged = root / 'data.zip'
    assembled = False
    if not merged.exists():
        parts = manifest['data_archive']['parts']
        if not all((root / part['name']).is_file() for part in parts):
            raise FileNotFoundError('Download data.zip from the repository release and put it next to reproduce.py')
        merged = root / '.data.zip'
        with merged.open('wb') as output:
            for part in parts:
                path = root / part['name']
                if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                    raise ValueError('Data volume differs: ' + part['name'])
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        output.write(block)
        assembled = True
    if digest(merged) != manifest['data_archive']['sha256']:
        raise ValueError('Combined data archive differs')
    extract_safe(merged, root)
    if assembled:
        merged.unlink()
    count = verify_files(root, manifest)
    return {'prepared': True, 'data_files': count, 'code_sha256': code['sha256']}


def verify_files(root, manifest=None):
    root = Path(root)
    manifest = manifest or read(root / 'release_manifest.json')
    for name, entry in manifest['files'].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Missing or changed data file: ' + name)
    with zipfile.ZipFile(root / manifest['code_archive']['name']) as archive:
        for name in archive.namelist():
            if (root / name).read_bytes() != archive.read(name):
                raise ValueError('Changed scientific source: ' + name)
    return len(manifest['files'])


def load_module(name):
    return importlib.import_module('experiments.aam_pricing.' + name)


def verify(root=ROOT):
    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    results = {'data_files_verified': verify_files(root)}

    finite = load_module('review11_finite_paths')
    finite_rows = read(root / 'results/trb_review11_finite_paths_v1/summary.json')['rows']
    designs = {r['id']: r['primitives'] for r in finite.design()['cases']}
    for row in finite_rows:
        finite.replay_case(designs[row['id']], row['result'])
    results['finite_welfare_paths'] = len(finite_rows)

    transition = load_module('review18_transition')
    folder = root / 'results/trb_review18_transition_v2'
    transition_fresh = transition.compute(read(folder / 'inputs.json'))
    assert_close(transition_fresh, read(folder / 'results.json'), 'fixed_quotation_transition')
    results['implementation_period_evaluations'] = len(transition_fresh['evaluations'])

    decomposition = load_module('review18_queue_decomposition')
    folder = root / 'results/trb_review18_queue_decomposition_v1'
    fresh = decomposition.compute(read(folder / 'inputs.json'))
    assert_close(fresh, read(folder / 'results.json'), 'queue_decomposition')
    results['fluid_queue_comparisons'] = len(fresh['comparisons'])
    results['saved_state_accounts'] = fresh['validated_saved_states']

    selection = load_module('review18_selection')
    fresh = selection.bundle()
    assert_close(fresh, read(root / 'results/review18_selection_v1/selection.json'), 'equilibrium_selection')
    results['exact_selection_example'] = True

    information = load_module('review17_information')
    folder = root / 'results/trb_review17_information_v1'
    plan = read(folder / 'design.json')
    true_saved = read(folder / 'true_menus.json')
    truth = {b['id']: information.rebuild_menu(b['case'], true_saved[b['id']])
             for b in plan['backgrounds']}
    beliefs = {b['id']: b for b in plan['beliefs']}
    saved = read(folder / 'belief_menus.json')
    rebuilt = [dict(b, rows=information.rebuild_menu(beliefs[b['id']]['case'], b['rows']))
               for b in saved]
    rebuilt_information = information.analyze(plan, truth, rebuilt)
    for name, value in rebuilt_information.items():
        assert_close(value, read(folder / (name + '.json')), 'input_information.' + name)
    results['forecast_belief_designs'] = len(saved)

    resources = load_module('review18_resources')
    results['independent_resource_check'] = resources.replay(root / 'results/trb_review18_resources_v1')

    from scripts import run_review18_capped_exact as witness
    actual = load_module('review18_capped_actual')
    folder = root / actual.OUTPUT
    cases = {r['id']: r['primitives'] for r in actual.design(root)['cases']}
    rows = read(folder / 'points.json')
    witnesses = read(folder / 'review18_capped_exact_witnesses.json')
    if len(rows) != 5 or len(witnesses) != 5:
        raise AssertionError('Expected five exact boundary witnesses')
    for row, bound in zip(rows, witnesses):
        witness.verify(cases[row['case_id']], row, bound)
    results['boundary_nash_witnesses'] = len(witnesses)

    complete = read(folder / 'review18_capped_actual_44_plot_ready.json')
    if len(complete) != 44:
        raise AssertionError('Expected a complete 44-response comparison')
    for row in complete:
        saved = row['state']
        fares = [44.0 if p is None else p for p in saved['prices']]
        current = actual.state(cases[row['case_id']], saved['charge'], fares, saved['active'])
        for name in ('rho', 'wait', 'resource_use', 'total_mass', 'TS'):
            assert_close(current[name], saved[name], row['case_id'] + '.' + name, atol=2e-6)
    results['fixed_fare_accounts_verified'] = len(complete)
    results['new_platform_games_solved'] = 0
    return results


def recompute(component, output, root=ROOT):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output path; saved evidence is never overwritten')
    if component == 'transition':
        module = load_module('review18_transition')
        value = module.compute(read(root / 'results/trb_review18_transition_v2/inputs.json'))
    elif component == 'queue-decomposition':
        module = load_module('review18_queue_decomposition')
        value = module.compute(read(root / 'results/trb_review18_queue_decomposition_v1/inputs.json'))
    else:
        value = load_module('review18_selection').bundle()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return {'component': component, 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare', help='Verify and unpack all code and data volumes')
    commands.add_parser('verify', help='Check supplied files and reproduce selected numerical evidence')
    fresh = commands.add_parser('recompute', help='Recalculate a component from supplied inputs')
    fresh.add_argument('component', choices=['transition', 'queue-decomposition', 'selection'])
    fresh.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare()
    elif args.command == 'verify':
        result = verify()
    else:
        result = recompute(args.component, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
