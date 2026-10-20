"""Frozen analytical participation economies and root-free identity replay."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import review11_finite_paths as fp

OUTPUT = 'results/trb_review16_participation_v1'
SOURCE_FILES = ('experiments/aam_pricing/review16_participation.py',
                'scripts/run_review16_participation.py', 'tests/test_review16_participation.py',
                'experiments/aam_pricing/review11_finite_paths.py',
                'experiments/aam_pricing/review8_resource.py',
                'experiments/aam_pricing/review7_structure.py',
                'experiments/__init__.py', 'experiments/aam_pricing/__init__.py')
KAPPAS = (0., .5, 1.)
TOL = 1e-9
COMPONENTS = ('margin', 'slack', 'gain_0', 'gain_05', 'gain_1', 'CS')


def referee_case():
    return dict(lam=.1, mu=.5, value=.4, capacity=60., resource=[.25,1.,2.5],
                costs=[9.,9.,9.], platform_quality=[4.5]*3, mass=[241.],
                group_quality=[0.], reliability_slopes=[.1], outside_utility=[1.7],
                outside_price=5., fare_bounds=[5.,120.])


def design():
    cases = []
    for N, U in product((241.,300.,480.,720.,1200.,1920.), (-1.,.5,1.7,3.,4.5)):
        c = referee_case()
        c.update(mass=[N], outside_utility=[U])
        cases.append(dict(id=f'N{N:g}_U{U:g}', primitives=c))
    return dict(schema_version=1, cases=cases, kappas=list(KAPPAS), initial_charge=0.,
                path_points=17, quadrature_orders=[8,16], quadrature_tolerance=TOL,
                quadrature_max_leaves=128, runtime_cap_seconds=600,
                primary_kappa=0., new_general_Nash_solves=0,
                interpretation='Designed analytical economies, not calibrated effects')


def effective_mass(case, rho):
    """Partition-invariant mass at fixed utilization; not a sampled global proof."""
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError('Utilization must be in [0,1]')
    z = (np.asarray(case['group_quality'])-rho*np.asarray(case['reliability_slopes'])
         -np.asarray(case['outside_utility']))/case['mu']
    return float(np.asarray(case['mass'])@np.exp(z-z.max()))


def enrich(case, state):
    theta = np.sum(state['probabilities'],axis=1)
    y, p, dy, dp, c = (np.asarray(v) for v in
        (state['demand'], state['prices'], state['demand_slope'],
         state['fare_slope'], case['costs']))
    h = case['lam']/case['mu']
    Y, L, dY = state['total_mass'], state['L'], state['total_mass_slope']
    return dict(state, theta=theta.tolist(), operating_margin=float((p-c)@y),
                margin_derivative=float(dp@y+(p-c)@dy),
                cs_derivative=Y*dY/(h*L), slack_derivative=(Y/L-1)*dY/h)


def _vector(case, state):
    return np.array([case['capacity']*state['fare_allocation_factor'],
                     state['slack_derivative'],
                     *(state['ts_derivatives'][str(k)] for k in KAPPAS),
                     state['cs_derivative']])


def adaptive_integral(case, stop, deadline=None):
    """Resolve each identity component, including small slack, separately."""
    rules = {str(n): dict(zip(('nodes','weights'),
             (v.tolist() for v in np.polynomial.legendre.leggauss(n)))) for n in (8,16)}

    def leaf(left,right):
        out = dict(left=float(left),right=float(right))
        integrals = []
        for label,n in (('low',8),('high',16)):
            rows = []
            for node,weight in zip(rules[str(n)]['nodes'],rules[str(n)]['weights']):
                fp._budget(deadline)
                q = (left+right)/2+(right-left)*node/2
                rows.append(dict(charge=q,weight=weight*(right-left)/2,
                                 state=enrich(case,fp.path_state(case,q,deadline=deadline))))
            out[label] = rows
            integrals.append(sum((r['weight']*_vector(case,r['state']) for r in rows),
                                 np.zeros(len(COMPONENTS))))
        out['integrals'] = integrals[1].tolist()
        out['error_estimates'] = abs(integrals[1]-integrals[0]).tolist()
        return out

    leaves = [leaf(0.,stop)]
    while True:
        values = np.sum([v['integrals'] for v in leaves],axis=0)
        errors = np.sum([v['error_estimates'] for v in leaves],axis=0)
        scale = TOL*np.maximum(1.,abs(values))
        if np.all(errors <= scale):
            break
        if len(leaves) >= 128:
            raise ValueError('Frozen quadrature budget exhausted')
        index = max(range(len(leaves)),key=lambda i: np.max(np.array(leaves[i]['error_estimates'])/scale))
        old = leaves.pop(index)
        mid = (old['left']+old['right'])/2
        leaves.extend((leaf(old['left'],mid),leaf(mid,old['right'])))
    return dict(rules=rules,leaves=sorted(leaves,key=lambda v:v['left']),
                components=list(COMPONENTS),integrals=values.tolist(),error_estimates=errors.tolist(),
                error_interpretation='Paired-rule estimates, not rigorous interval enclosures')


def bounds(case, states, quadrature):
    a,b = states[0],states[-1]
    deltaY = b['total_mass']-a['total_mass']
    deltaM = b['operating_margin']-a['operating_margin']
    h = case['lam']/case['mu']
    integ = quadrature['integrals']
    closed = None
    if len(case['mass']) == 1:
        N = case['mass'][0]
        closed = float((-N*np.log1p(-deltaY/(N-a['total_mass']))-deltaY)/h)
    rows = []
    for i,k in enumerate(KAPPAS):
        gain = b['TS'][str(k)]-a['TS'][str(k)]
        expansion = (1/h-k*case['outside_price'])*deltaY
        bound = deltaM+expansion
        rows.append(dict(kappa=k,finite_gain=gain,lower_bound=bound,tightness=bound/gain,
                         slack=gain-bound,margin_change=deltaM,expansion_bound=expansion,
                         deltaY=deltaY,integrated_margin=integ[0],integrated_slack=integ[1],
                         integrated_gain=integ[2+i],integrated_CS=integ[5],
                         CS_change=b['net_surplus']-a['net_surplus'],
                         bound_from_integral=integ[0]+expansion,
                         closed_single_group_slack=closed))
    return rows


def evaluate_case(case, deadline=None):
    fp._budget(deadline)
    try:
        fp.validate(case,KAPPAS)
    except ValueError as e:
        return dict(status='ineligible',reasons=['primitive premise: '+str(e)],
                    qualifying_release=False,bounds=[],new_general_Nash_solves=0)
    ends = fp.locate_endpoints(case,deadline)
    if not ends['eligible']:
        return dict(status='ineligible',reasons=ends['reasons'],endpoints=ends,
                    qualifying_release=False,bounds=[],new_general_Nash_solves=0)
    kind = fp._classification(ends)
    stop = min(ends['release_charge'],ends['fare_charge'])
    endpoint = {'release_first':'release','fare_first':'fare','coincident':'coincident'}[kind]
    states = [enrich(case,fp.path_state(case,float(q),endpoint=endpoint if i == 16 else None,
                                      deadline=deadline))
              for i,q in enumerate(np.linspace(0.,stop,17))]
    quad = adaptive_integral(case,stop,deadline)
    r = dict(status='completed',classification=kind,qualifying_release=kind=='release_first',
             stop_charge=stop,endpoints=ends,states=states,quadrature=quad,
             bounds=bounds(case,states,quad),new_general_Nash_solves=0,
             numerical_global_Nash_proof=False,
             equilibrium_basis='Analytical full-capacity profit domination; original sufficient uniqueness premises')
    r['validation'] = replay_case(case,r)
    return r


def _same(saved, expected, label):
    if isinstance(expected,dict):
        if set(saved) != set(expected):
            raise ValueError('Saved keys differ: '+label)
        for k,v in expected.items():
            _same(saved[k],v,label+':'+k)
    elif isinstance(expected,(list,tuple)):
        if len(saved) != len(expected):
            raise ValueError('Saved sequence length differs: '+label)
        for i,(a,b) in enumerate(zip(saved,expected)):
            _same(a,b,label+':'+str(i))
    elif expected is None or isinstance(expected,(str,bool)):
        if saved != expected:
            raise ValueError('Saved value differs: '+label)
    else:
        fp._close(saved,expected,label)


def _replay_state(case,saved):
    base = fp._replay_state(case,saved)
    base.update(fp._zero_wait_witness(case,base))
    rebuilt = enrich(case,base)
    _same(saved,rebuilt,'state')
    fp._close(rebuilt['margin_derivative'],case['capacity']*rebuilt['fare_allocation_factor'],
              'margin derivative identity')
    fp._close(rebuilt['total_mass']-rebuilt['L'],
              np.asarray(case['mass'])@np.square(rebuilt['theta']),'Y-L identity')
    return rebuilt


def replay_case(case,r):
    """Substitute saved roots and Gaussian nodes; never locate roots again."""
    if r['status'] == 'ineligible':
        if 'endpoints' not in r:
            try:
                fp.validate(case,KAPPAS)
            except ValueError as e:
                _same(r['reasons'],['primitive premise: '+str(e)],'premise exclusion')
            else:
                raise ValueError('Unsupported primitive exclusion')
        else:
            fp.validate(case,KAPPAS)
            fp._replay_auxiliary(case,r['endpoints']['initial'])
            reasons = fp._ineligible_reasons(case,r['endpoints']['initial'])
            if not reasons or r['endpoints']['eligible']:
                raise ValueError('Unsupported initial exclusion')
            _same(r['reasons'],reasons,'initial exclusion')
        if r['qualifying_release'] or r['bounds'] or r['new_general_Nash_solves'] != 0:
            raise ValueError('Ineligible case included in guarantee')
        return dict(valid=True,status='ineligible',scalar_roots_called=0)
    fp.validate(case,KAPPAS)
    if r['status'] != 'completed':
        raise ValueError('Incomplete case')
    ends = r['endpoints']
    for k in ('initial','release','fare'):
        fp._replay_auxiliary(case,ends[k])
    fp._close(ends['initial']['charge'],0.,'initial charge')
    if not ends['eligible'] or fp._ineligible_reasons(case,ends['initial']):
        raise ValueError('Invalid initial premise')
    for k in ('release','fare'):
        fp._close(ends[k]['charge'],ends[k+'_charge'],k+' charge')
    fp._close(ends['release']['zero_wait_excess'],0.,'release residual')
    fp._close(max(ends['fare']['prices']),case['fare_bounds'][1],'fare residual')
    kind = fp._classification(ends)
    if r['classification'] != kind or r['qualifying_release'] != (kind=='release_first'):
        raise ValueError('Endpoint classification differs')
    stop = min(ends['release_charge'],ends['fare_charge'])
    fp._close(r['stop_charge'],stop,'stop charge')
    states = r['states']
    if len(states) != 17:
        raise ValueError('Incomplete path states')
    fp._close([s['charge'] for s in states],np.linspace(0.,stop,17),'path nodes')
    for s in states:
        _replay_state(case,s)
        if min(s['prices']) <= case['fare_bounds'][0] or max(s['prices']) > case['fare_bounds'][1]+1e-8:
            raise ValueError('Fare domain violated')
    if any(s['wait'] <= 0 for s in states[:-1]):
        raise ValueError('Premature release')
    if kind == 'release_first':
        if states[-1]['wait'] != 0 or max(states[-1]['prices']) >= case['fare_bounds'][1]:
            raise ValueError('Noninterior release endpoint')
    else:
        fp._close(max(states[-1]['prices']),case['fare_bounds'][1],'boundary stop')
    q = r['quadrature']
    _same(q['components'],list(COMPONENTS),'integral components')
    left = 0.
    for leaf in q['leaves']:
        fp._close(leaf['left'],left,'leaf adjacency',1e-11)
        right = leaf['right']
        if right <= left:
            raise ValueError('Nonpositive quadrature interval')
        values = []
        for label,n in (('low',8),('high',16)):
            nodes,weights = np.polynomial.legendre.leggauss(n)
            _same(q['rules'][str(n)],dict(nodes=nodes.tolist(),weights=weights.tolist()),'Gaussian rule')
            rows = leaf[label]
            if len(rows) != n:
                raise ValueError('Incomplete Gaussian rule')
            for row,node,weight in zip(rows,nodes,weights):
                fp._close(row['charge'],(left+right)/2+(right-left)*node/2,'Gaussian node',1e-11)
                fp._close(row['weight'],weight*(right-left)/2,'Gaussian weight',1e-11)
                fp._close(row['state']['charge'],row['charge'],'state-node match',1e-11)
                _replay_state(case,row['state'])
                if row['state']['wait'] <= 0:
                    raise ValueError('Quadrature outside strict branch')
            values.append(sum((row['weight']*_vector(case,row['state']) for row in rows),
                              np.zeros(len(COMPONENTS))))
        _same(leaf['integrals'],values[1],'leaf integrals')
        _same(leaf['error_estimates'],abs(values[1]-values[0]),'leaf errors')
        left = right
    fp._close(left,stop,'quadrature coverage',1e-11)
    vals = np.sum([v['integrals'] for v in q['leaves']],axis=0)
    errors = np.sum([v['error_estimates'] for v in q['leaves']],axis=0)
    _same(q['integrals'],vals,'total integrals')
    _same(q['error_estimates'],errors,'total errors')
    if len(q['leaves']) > 128 or np.any(errors > TOL*np.maximum(1.,abs(vals))):
        raise ValueError('Quadrature tolerance failure')
    expected = bounds(case,states,q)
    if len(r['bounds']) != len(KAPPAS):
        raise ValueError('Incomplete accounting variants')
    for saved,b in zip(r['bounds'],expected):
        _same(saved,b,'bounds')
        for actual,want,label in ((b['finite_gain'],b['integrated_gain'],'TS derivative'),
            (b['slack'],b['integrated_slack'],'exact slack'),
            (b['margin_change'],b['integrated_margin'],'margin endpoint'),
            (b['CS_change'],b['integrated_CS'],'CS integral')):
            fp._close(actual,want,label,2e-6)
        if b['closed_single_group_slack'] is not None:
            fp._close(b['slack'],b['closed_single_group_slack'],'closed logarithmic slack',2e-6)
        if b['lower_bound'] <= 0 or b['slack'] < -2e-7:
            raise ValueError('Finite bound failed')
    if r['new_general_Nash_solves'] != 0 or r['numerical_global_Nash_proof']:
        raise ValueError('Unsupported Nash computation claim')
    return dict(valid=True,status='completed',scalar_roots_called=0,
                qualifying_release=r['qualifying_release'],
                maximum_identity_error=max(abs(b['slack']-b['integrated_slack']) for b in expected),
                maximum_margin_error=abs(expected[0]['margin_change']-expected[0]['integrated_margin']),
                release_residual=abs(ends['release']['zero_wait_excess']),
                minimum_fare_clearance=min(min(s['prices'])-case['fare_bounds'][0] for s in states),
                terminal_upper_fare_clearance=case['fare_bounds'][1]-max(states[-1]['prices']))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def prepare(root,output):
    root,output = Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'inputs.json').exists() or (output/'execution_manifest.json').exists():
        raise FileExistsError('Frozen preparation exists; never overwrite')
    if not (output/'design_freeze.md').exists():
        raise FileNotFoundError('Freeze design before production')
    inputs = design()
    _dump(output/'inputs.json',inputs)
    for name in SOURCE_FILES:
        dest = output/'execution_source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,dest)
    references = ()
    for p in references:
        dest = output/'reference_source'/p.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
    files = [p for p in output.rglob('*') if p.is_file()]
    manifest = dict(files={p.relative_to(output).as_posix():_sha(p) for p in files},
                    workspace_sources={name:_sha(root/name) for name in SOURCE_FILES},
                    design_frozen_before_execution=True)
    _dump(output/'execution_manifest.json',manifest)
    return dict(prepared=True,cases=len(inputs['cases']),input_sha256=_sha(output/'inputs.json'))


def _verify_manifest(output,root=None):
    m = json.loads((output/'execution_manifest.json').read_text(encoding='utf-8'))
    for name,digest in m['files'].items():
        if _sha(output/name) != digest:
            raise ValueError('Frozen file changed: '+name)
    if root is not None:
        for name,digest in m['workspace_sources'].items():
            if _sha(root/name) != digest:
                raise ValueError('Execution source changed: '+name)
    return m


def table_rows(cases):
    rows = []
    for row in cases:
        c,r = row['primitives'],row['result']
        base = dict(id=row['id'],population=sum(c['mass']),capacity=c['capacity'],
                    population_per_capacity=sum(c['mass'])/c['capacity'],
                    normalized_population=sum(c['mass'])*min(c['resource'])/c['capacity'],
                    outside_utility=c['outside_utility'][0],status=r['status'],
                    qualifying_release=r['qualifying_release'],
                    classification=r.get('classification',''),reasons='; '.join(r.get('reasons',[])))
        if r['status'] != 'completed':
            rows.append(base)
            continue
        theta = np.array([s['theta'] for s in r['states']])
        base.update(release_charge=r['endpoints']['release_charge'],fare_charge=r['endpoints']['fare_charge'],
                    stop_charge=r['stop_charge'],theta_initial_min=min(theta[0]),
                    theta_initial_max=max(theta[0]),theta_terminal_min=min(theta[-1]),
                    theta_terminal_max=max(theta[-1]),theta_path_min=float(theta.min()),
                    theta_path_max=float(theta.max()),wait_initial=r['states'][0]['wait'],
                    wait_terminal=r['states'][-1]['wait'],
                    terminal_max_fare=max(r['states'][-1]['prices']),
                    quadrature_leaves=len(r['quadrature']['leaves']),
                    quadrature_margin_error=r['quadrature']['error_estimates'][0],
                    quadrature_slack_error=r['quadrature']['error_estimates'][1],
                    **{k:v for k,v in r['validation'].items() if k not in ('status','valid','qualifying_release')})
        for b in r['bounds']:
            rows.append(dict(base,**b))
    return rows


def _csv(path,rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def path_rows(cases):
    rows = []
    for row in cases:
        for s in row['result'].get('states',[]):
            rows.append(dict(id=row['id'],charge=s['charge'],theta=s['theta'][0],Y=s['total_mass'],
                             L=s['L'],Y_over_L=s['total_mass']/s['L'],Y_prime=s['total_mass_slope'],
                             wait=s['wait'],margin=s['operating_margin'],
                             margin_derivative=s['margin_derivative'],slack_derivative=s['slack_derivative'],
                             max_fare=max(s['prices']),qualifying_release=row['result']['qualifying_release']))
    return rows


def _summary(cases):
    eligible = [v for v in cases if v['result']['qualifying_release']]
    primary = [v['result']['bounds'][0] for v in eligible]
    return dict(economies=len(cases),qualifying_release=len(eligible),
                ineligible=sum(v['result']['status']=='ineligible' for v in cases),
                fare_first=sum(v['result'].get('classification')=='fare_first' for v in cases),
                failed=sum(v['result']['status']=='failed' for v in cases),
                primary_coverage_range=[min(b['tightness'] for b in primary),max(b['tightness'] for b in primary)] if primary else [],
                participation_range=[min(s['theta'][0] for v in eligible for s in v['result']['states']),
                                     max(s['theta'][0] for v in eligible for s in v['result']['states'])] if primary else [],
                new_general_Nash_solves=0,post_release_policy_claim=False)


def run(root,output):
    root,output = Path(root),Path(output)
    _verify_manifest(output,root)
    _dump(output/'attempt.json',dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                   rerun_allowed=False))
    started = time.monotonic()
    deadline = started+600.
    inputs = json.loads((output/'inputs.json').read_text(encoding='utf-8'))
    cases = []
    for row in inputs['cases']:
        try:
            result = evaluate_case(row['primitives'],deadline)
        except Exception as e:
            result = dict(status='failed',qualifying_release=False,reasons=[type(e).__name__+': '+str(e)])
        saved = dict(deepcopy(row),result=result)
        cases.append(saved)
        _dump(output/(row['id']+'.json'),saved)
        if time.monotonic() >= deadline:
            break
    result = dict(cases=cases,summary=_summary(cases),elapsed_seconds=time.monotonic()-started,
                  inputs_sha256=_sha(output/'inputs.json'))
    _verify_manifest(output,root)
    _dump(output/'results.json',result)
    _csv(output/'economies.csv',table_rows(cases))
    _csv(output/'path_states.csv',path_rows(cases))
    return result


def replay(output):
    output = Path(output)
    _verify_manifest(output)
    inputs = json.loads((output/'inputs.json').read_text(encoding='utf-8'))
    r = json.loads((output/'results.json').read_text(encoding='utf-8'))
    if r['inputs_sha256'] != _sha(output/'inputs.json') or len(r['cases']) != len(inputs['cases']):
        raise ValueError('Incomplete execution or changed inputs')
    checks = []
    for source,row in zip(inputs['cases'],r['cases']):
        _same(dict(id=row['id'],primitives=row['primitives']),source,'case inputs')
        saved = json.loads((output/(row['id']+'.json')).read_text(encoding='utf-8'))
        if saved != row:
            raise ValueError('Case file differs from aggregate results')
        checks.append(replay_case(row['primitives'],row['result']))
    _same(r['summary'],_summary(r['cases']),'summary')
    for name,expected in (('economies.csv',table_rows(r['cases'])),
                          ('path_states.csv',path_rows(r['cases']))):
        with (output/name).open(newline='',encoding='utf-8') as f:
            actual = list(csv.DictReader(f))
        if len(actual) != len(expected):
            raise ValueError('CSV row mismatch: '+name)
        for a,b in zip(actual,expected):
            for k,v in b.items():
                if a[k] != ('' if v is None else str(v)):
                    raise ValueError('CSV arithmetic mismatch: '+name+':'+k)
    return dict(valid=all(v['valid'] for v in checks),cases_validated=len(checks),
                scalar_roots_called=0,adaptive_integral_calls=0,writes=0,summary=r['summary'])
