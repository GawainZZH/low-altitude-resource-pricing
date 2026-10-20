"""Transfer-free symmetric endpoint decomposition of saved fluid/queue menus.

Only arithmetic at saved states: no equilibrium or queue-closure imports.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

SOURCE = 'results/trb_review7_queue_v1'
INPUTS = (SOURCE+'/static_reference_inputs.json', SOURCE+'/raw.json')
OWNED = ('experiments/aam_pricing/review18_queue_decomposition.py',
         'scripts/run_review18_queue_decomposition.py',
         'tests/test_review18_queue_decomposition.py')
CHARGES = [0., 4., 8., 12., 16., 20.]
COMPONENTS = ('waiting_relief', 'reliability_relief', 'choice_allocation',
              'operating_cost_saving', 'system_cost_saving')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload(root):
    root = Path(root)
    fluid, queue = [json.loads((root/p).read_text(encoding='utf-8')) for p in INPUTS]
    cases = {c['name']: c for c in fluid['cases']}
    states = []
    for row in fluid['historical_games']:
        c = next(c for c in cases.values() if
                 [c[k] for k in ('demand_scale', 'capacity_scale', 'reliability_power')] == row['profile'])
        s = row['waiting_outcome']
        states.append(dict(profile=c['name'], institution='fluid', charge=row['charge'],
                           prices=row['selected']['prices'], rho=s['rho'],
                           wait_minutes=s['wait_minutes'], demand=s['demand'], cs=s['cs']))
    for row in queue['games']:
        if row['profile'] in cases and row['accepted']:
            states.append(dict(institution='queue', **{k: row[k] for k in
                ('profile', 'charge', 'prices', 'rho', 'wait_minutes', 'demand', 'cs',
                 'ts', 'probabilities', 'profits', 'resource_receipts', 'fare_receipts',
                 'real_cost', 'system_cost')}))
    return dict(cases=cases, states=states, common_menu=CHARGES,
                externality=2800., ground_margin_credit=0., conditional_service_cv2=1.)


def account(c, s):
    mu = c['logit_scale']
    rho, wait, fares = s['rho'], s['wait_minutes'], s['prices']
    if not 0 <= rho <= 1 or wait < 0 or not all(math.isfinite(x) for x in [rho, wait, *fares]):
        raise ValueError('Finite feasible saved state required')
    masses = [d*c['demand_scale'] for d in c['base_demand']]
    loss_factor = .55*rho**c['reliability_power']+1.10*max(rho-.88, 0.)**2
    loss = [b+e*loss_factor for b, e in zip(c['base_loss'], c['exposure'])]
    costs = [c['base_cost'][j]+[0., c['cost_gap'], .5*c['cost_gap']][j] for j in range(3)]
    probs, group_flow, logsum_cs = [], [], []
    intrinsic, entropy, time_mass, reliability_mass = [], [], [], [[], [], []]
    for g, d in enumerate(masses):
        lam, beta, gamma = [c[k][g] for k in
            ('price_sensitivity', 'time_sensitivity', 'reliability_sensitivity')]
        base = [c['platform_affinity'][g][j]+(c['differentiation'] if j == g else 0.)
                -beta*c['service_time'][j] for j in range(3)]
        outside = c['outside_affinity'][g]-lam*c['outside_price']-beta*c['outside_time']
        u = [(base[j]-lam*fares[j]-beta*wait-gamma*loss[j])/mu for j in range(3)]+[outside/mu]
        peak = max(u)
        logden = peak+math.log(math.fsum(math.exp(v-peak) for v in u))
        p = [math.exp(v-logden) for v in u]
        probs.append(p)
        group_flow.append([d*z for z in p])
        logsum_cs.append(d*mu/lam*logden)
        intrinsic.append(d/lam*math.fsum(z*b for z, b in zip(p, base+[outside])))
        entropy.append(-d*mu/lam*math.fsum(z*math.log(z) for z in p if z))
        time_mass.append(d*beta/lam*sum(p[:3]))
        for j in range(3):
            reliability_mass[j].append(d*gamma/lam*p[j])
    y = [math.fsum(q[j] for q in group_flow) for j in range(3)]
    T = math.fsum(time_mass)
    B = [math.fsum(z) for z in reliability_mass]
    gross = math.fsum(intrinsic)+math.fsum(entropy)
    waiting_cost, reliability_cost = wait*T, math.fsum(l*b for l, b in zip(loss, B))
    revenue = math.fsum(p*n for p, n in zip(fares, y))
    operating = math.fsum(c*n for c, n in zip(costs, y))
    system = c['congestion_cost']*(.15*rho*rho+max(rho-.90, 0.)**3)
    work = math.fsum(a*n for a, n in zip(c['platform_resource'], y))
    receipts = s['charge']*work
    profits = revenue-operating-receipts
    cs = math.fsum(logsum_cs)
    ts = cs+revenue-operating-system
    demand_error = max(abs(a-b) for a, b in zip(y, s['demand']))
    cs_error = abs(cs-s['cs'])
    if demand_error > 1e-7 or cs_error > 1e-7:
        raise ValueError('Saved choice arithmetic mismatch')
    if 'ts' in s and abs(ts-s['ts']) > 1e-7:
        raise ValueError('Saved TS arithmetic mismatch')
    saved_error = 0.
    if 'probabilities' in s:
        saved_error = max(abs(probs[g][j]-s['probabilities'][g][j]) for g in range(3) for j in range(3))
        for key, value in [('fare_receipts', revenue), ('resource_receipts', receipts),
                           ('real_cost', operating), ('system_cost', system)]:
            saved_error = max(saved_error, abs(s[key]-value))
        saved_error = max(saved_error, abs(sum(s['profits'])-profits))
        if saved_error > 1e-7:
            raise ValueError('Saved account arithmetic mismatch')
    return dict(prices=fares, charge=s['charge'], rho=rho, wait_minutes=wait,
        probabilities=probs, group_flows=group_flow, demand=y, passengers=math.fsum(y),
        outside_passengers=math.fsum(q[3] for q in group_flow), work=work,
        capacity=c['base_capacity']*c['capacity_scale'], CS=cs, TS=ts,
        intrinsic_choice=math.fsum(intrinsic), entropy_choice=math.fsum(entropy),
        gross_choice=gross, time_value_mass=T, reliability_value_mass=B, loss=loss,
        waiting_expense=waiting_cost, reliability_expense=reliability_cost,
        operating_cost=operating, system_cost=system, fare_receipts=revenue,
        platform_profit=profits, resource_receipts=receipts,
        logsum_identity_error=cs+revenue-(gross-waiting_cost-reliability_cost),
        transfer_identity_error=ts-(cs+profits+receipts-system),
        saved_demand_error=demand_error, saved_cs_error=cs_error, saved_account_error=saved_error)


def decompose(a, b):
    dw = b['wait_minutes']-a['wait_minutes']
    dt = b['time_value_mass']-a['time_value_mass']
    dl = [y-x for x, y in zip(a['loss'], b['loss'])]
    db = [y-x for x, y in zip(a['reliability_value_mass'], b['reliability_value_mass'])]
    avg_t = (a['time_value_mass']+b['time_value_mass'])/2
    avg_w = (a['wait_minutes']+b['wait_minutes'])/2
    avg_b = [(x+y)/2 for x, y in zip(a['reliability_value_mass'], b['reliability_value_mass'])]
    avg_l = [(x+y)/2 for x, y in zip(a['loss'], b['loss'])]
    detail = dict(intrinsic_change=b['intrinsic_choice']-a['intrinsic_choice'],
                  entropy_change=b['entropy_choice']-a['entropy_choice'],
                  waiting_composition=-avg_w*dt,
                  reliability_composition=-math.fsum(x*y for x, y in zip(avg_l, db)))
    components = dict(waiting_relief=-dw*avg_t,
        reliability_relief=-math.fsum(x*y for x, y in zip(dl, avg_b)),
        choice_allocation=math.fsum(detail.values()),
        operating_cost_saving=a['operating_cost']-b['operating_cost'],
        system_cost_saving=a['system_cost']-b['system_cost'])
    paths = {}
    for name, env, q in [('choices_first', a, b), ('environment_first', b, a)]:
        paths[name] = dict(choice=b['gross_choice']-a['gross_choice']-env['wait_minutes']*dt
            -math.fsum(l*v for l, v in zip(env['loss'], db)),
            waiting=-dw*q['time_value_mass'],
            reliability=-math.fsum(l*v for l, v in zip(dl, q['reliability_value_mass'])),
            operating=components['operating_cost_saving'], system=components['system_cost_saving'])
    gain = b['TS']-a['TS']
    error = math.fsum(components.values())-gain
    if abs(error) > 1e-7 or any(abs(math.fsum(p.values())-gain)>1e-7 for p in paths.values()):
        raise ValueError('Endpoint identity mismatch')
    return dict(components=components, choice_detail=detail, order_paths=paths,
                gain=gain, accounting_error=error)


def compute(data):
    if data['common_menu'] != CHARGES or len(data['states']) != 24:
        raise ValueError('Exactly two backgrounds, two institutions, six saved charges required')
    rows, all_accounts, branch = [], [], []
    for name, c in data['cases'].items():
        for institution in ('fluid', 'queue'):
            saved = sorted([s for s in data['states'] if s['profile'] == name
                            and s['institution'] == institution], key=lambda x: x['charge'])
            if [s['charge'] for s in saved] != CHARGES:
                raise ValueError('Incomplete or duplicate saved menu')
            accounts = [account(c, s) for s in saved]
            all_accounts.extend(accounts)
            # Maximize the existing common menu only; never solve a new game.
            best = max(accounts, key=lambda a: (a['TS'], -a['charge']))
            nc = accounts[0]
            if institution == 'fluid':
                last = max((a for a in accounts if a['wait_minutes'] > 1e-7), key=lambda a: a['charge'])
                branch.append(dict(profile=name, last_positive_wait_charge=last['charge'],
                    NC_passengers=nc['passengers'], last_positive_wait_passengers=last['passengers'],
                    passengers_change=last['passengers']-nc['passengers'],
                    work_change=last['work']-nc['work'], wait_minutes=last['wait_minutes'],
                    selected_menu_optimum_is_full_work=abs(best['work']-best['capacity']) < 1e-7))
            rows.append(dict(profile=name, institution=institution, selected_charge=best['charge'],
                selected_at_upper_menu_boundary=best['charge'] == CHARGES[-1],
                NC=nc, selected=best, passengers_change=best['passengers']-nc['passengers'],
                work_change=best['work']-nc['work'], **decompose(nc, best)))
    max_error = max(abs(a[k]) for a in all_accounts for k in
                    ('logsum_identity_error', 'transfer_identity_error', 'saved_demand_error',
                     'saved_cs_error', 'saved_account_error'))
    return dict(method='exact symmetric endpoint order-average, transfer-free entropy identity',
        common_menu=CHARGES, H_minutes=120, externality=2800., ground_margin_credit=0.,
        conditional_service_cv2=1., comparisons=rows, fluid_branch_diagnostics=branch,
        validated_saved_states=len(all_accounts),
        max_account_error=max_error, new_Nash_solves=0, new_behavioral_closures=0,
        new_queue_variants=0, new_figures=0,
        interpretation='Choice allocation includes all endogenous probability changes; not a causal fare-only effect. Hybrids are accounting evaluations, not equilibria. Each institution retains its own NC.')


def write_results(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    hashes = {p: digest(root/p) for p in INPUTS}
    data = payload(root)
    r = compute(data)
    output.mkdir(parents=True)
    for filename, obj in [('inputs.json', data), ('results.json', r)]:
        (output/filename).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    table = []
    for x in r['comparisons']:
        table.append(dict(profile=x['profile'], institution=x['institution'],
            selected_charge=x['selected_charge'], NC_TS=x['NC']['TS'], selected_TS=x['selected']['TS'],
            gain=x['gain'], NC_passengers=x['NC']['passengers'], selected_passengers=x['selected']['passengers'],
            NC_wait=x['NC']['wait_minutes'], selected_wait=x['selected']['wait_minutes'],
            NC_work=x['NC']['work'], selected_work=x['selected']['work'],
            **x['components'], accounting_error=x['accounting_error']))
    with (output/'comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for p in OWNED:
        target = output/'execution_source'/p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/p, target)
    if hashes != {p: digest(root/p) for p in INPUTS}:
        raise ValueError('Old input changed during arithmetic generation')
    manifest = dict(old_sources=hashes, old_sources_unchanged=True,
        files={str(p.relative_to(output)).replace('\\', '/'): digest(p)
               for p in output.rglob('*') if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return r


def replay(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for p, expected in manifest['files'].items():
        if digest(folder/p) != expected:
            raise ValueError('Frozen hash mismatch: '+p)
    data = json.loads((folder/'inputs.json').read_text(encoding='utf-8'))
    if compute(data) != json.loads((folder/'results.json').read_text(encoding='utf-8')):
        raise ValueError('Frozen arithmetic replay mismatch')
    return dict(verified=True, saved_states=24, comparisons=4, writes=0,
                new_Nash_solves=0, new_behavioral_closures=0, new_queue_variants=0)
