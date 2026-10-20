"""Task C: closed arithmetic on saved responses, with no behavioral solvers."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

ACCESS = 'results/trb_review6_access_v1/saved_inputs.json'
QUEUE = 'results/trb_review8_queue_variability_v1/raw.json'
EXTENSION = 'results/trb_review10_queue_endpoint_v1/raw.json'
DESIGN = 'docs/qa/trb_thirteenth_review_20261018/institution_comparison_design.md'
OWNED = ('experiments/aam_pricing/review13_institution.py',
         'scripts/run_review13_institution.py', 'tests/test_review13_institution.py')
PINNED = {
    ACCESS: 'f69095aa1a73859ade9895fa3705e412715184608ad2bba2be0949e1e3167020',
    QUEUE: '8e2efd649335e7d6af8b6af71186e16f29183597f069a02cef88a98935f4640c',
    EXTENSION: '185eb4099a768bf59a112808e69e4083e5d76a0a156f57c5ea2335698ad194dd',
}
REFERENCES = (
    'results/trb_review6_access_v1/policies.csv',
    'results/trb_review6_access_v1/summary.json',
    'results/trb_review8_queue_variability_v1/policies.csv',
    'results/trb_review9_queue_diagnostics_v1/summary.json',
    'results/trb_review10_queue_endpoint_v1/policies.csv',
    'results/trb_review10_queue_endpoint_v1/summary.json',
)
PROFILES = ((.85, 1.12, 2.2), (1.35, .9, 2.2), (1.7, .7, 2.2))
ES = (90., 1200., 2800., 4200., 6500.)
KAPPAS = (0., .5, 1.)
MENU = list(range(0, 22, 2))
COMMON = [0, 4, 8, 12, 16, 20]
EXPANDED_PROFILE = 'grid_d170_c070_p170_e2800'
INSTITUTIONS = ('real_waiting', 'transferable_payment')
METRICS = ('TS', 'CS', 'air_passengers', 'resource_work', 'physical_wait_minutes',
           'passenger_wait_minutes', 'real_access_cost', 'fare_payment',
           'access_payment', 'traveler_payment', 'platform_profit',
           'public_charge_revenue', 'allocator_rent_revenue',
           'operating_resource_cost', 'external_cost', 'equivalent_access_minutes')


def required(obj, names):
    if not isinstance(obj, dict) or any(k not in obj for k in names):
        raise ValueError('Unrecognized/incomplete source fields: '+', '.join(names))


def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('Finite numeric source field required')
    return float(x)


def vector(x, n=3):
    if not isinstance(x, list) or len(x) != n:
        raise ValueError('Source vector dimension differs')
    return [number(v) for v in x]


def close(a, b, label='saved arithmetic', atol=1e-8):
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            raise ValueError(label+': dimensions differ')
        for x, y in zip(a, b):
            close(x, y, label, atol)
    elif abs(number(a)-number(b)) > atol:
        raise ValueError(label+': values differ')


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Dot product dimensions differ')
    return sum(x*y for x, y in zip(a, b))


def effective(record):
    required(record, ('dataclass', 'effective'))
    c, e = record['dataclass'], record['effective']
    required(c, ('name', 'demand_scale', 'capacity_scale', 'reliability_power',
                 'congestion_cost', 'price_lower', 'price_upper'))
    required(e, ('D', 'alpha', 'alpha_outside', 'lambda', 'beta', 'gamma', 'mu',
                 'c', 'chi', 'ell', 'a', 'K', 'service_time', 'outside_price',
                 'outside_time', 'price_domain'))
    for k in ('D', 'alpha_outside', 'lambda', 'beta', 'gamma', 'c', 'chi', 'ell', 'a', 'service_time'):
        vector(e[k])
    if not isinstance(e['alpha'], list) or len(e['alpha']) != 3:
        raise ValueError('Source utility matrix dimension differs')
    for row in e['alpha']:
        vector(row)
    for k in ('mu', 'K', 'outside_price', 'outside_time'):
        number(e[k])
    for k in ('demand_scale', 'capacity_scale', 'reliability_power', 'congestion_cost', 'price_lower', 'price_upper'):
        number(c[k])
    if min(e['D']+e['lambda']+e['a']+[e['mu'], e['K'], c['reliability_power']]) <= 0:
        raise ValueError('Positive masses, sensitivities, resource and scales required')
    if min(e['beta']+e['gamma']+e['c']+e['chi']+e['ell']+[c['congestion_cost']]) < 0:
        raise ValueError('Negative cost or sensitivity')
    close(e['price_domain'], [c['price_lower'], c['price_upper']], 'fare domain')
    return c, e


def choice(e, power, prices, rho, wait, access_beta):
    """A single direct utility calculation at the saved rho and wait."""
    h = .55*rho**power+1.1*max(rho-.88, 0.)**2
    loss = [ell+chi*h for ell, chi in zip(e['ell'], e['chi'])]
    probabilities, cs_groups = [], []
    for g in range(3):
        lam, beta, gamma = (e[k][g] for k in ('lambda', 'beta', 'gamma'))
        u = [(e['alpha'][g][j]-lam*prices[j]-beta*e['service_time'][j]
              -gamma*loss[j]-access_beta[g]*wait)/e['mu'] for j in range(3)]
        outside = (e['alpha_outside'][g]-lam*e['outside_price']-beta*e['outside_time'])/e['mu']
        peak = max(u+[outside])
        den = peak+math.log(sum(math.exp(x-peak) for x in u+[outside]))
        probabilities.append([math.exp(x-den) for x in u])
        cs_groups.append(e['D'][g]*e['mu']/lam*den)
    y = [sum(e['D'][g]*probabilities[g][j] for g in range(3)) for j in range(3)]
    return y, probabilities, sum(cs_groups)


def reconstruct(record, game, institution):
    if institution not in INSTITUTIONS:
        raise ValueError('Unknown allocation institution')
    c, e = effective(record)
    required(game, ('profile', 'charge', 'mode', 'selected', 'waiting_outcome'))
    required(game['selected'], ('prices', 'verified', 'br'))
    s = game['waiting_outcome']
    required(s, ('wait_minutes', 'rho', 'demand', 'group_air_probability', 'cs',
                 'closure', 'complementarity'))
    profile = [c[k] for k in ('demand_scale', 'capacity_scale', 'reliability_power')]
    if game['mode'] != 'equal_vot' or vector(game['profile']) != profile:
        raise ValueError('Not the matched equal-money-value profile')
    if game['selected']['verified'] is not True:
        raise ValueError('Saved response was not accepted')
    p = vector(game['selected']['prices']); q = number(game['charge'])
    rho, wait = number(s['rho']), number(s['wait_minutes'])
    if q not in MENU or not 0 <= rho <= 1 or wait < 0:
        raise ValueError('Invalid saved charge/utilization/access expense')
    if min(p) < c['price_lower'] or max(p) > c['price_upper']:
        raise ValueError('Saved fare outside box')
    v = dot(e['D'], e['beta'])/dot(e['D'], e['lambda'])
    y, prob, cs = choice(e, c['reliability_power'], p, rho, wait,
                         [lam*v for lam in e['lambda']])
    close(y, s['demand'], 'saved demand')
    close([sum(row) for row in prob], s['group_air_probability'], 'group participation')
    close(cs, s['cs'], 'saved CS')
    work = dot(e['a'], y)
    close(work/e['K'], rho, 'resource closure', 1e-9)
    close(wait*(1-rho), 0., 'complementarity')
    close(s['closure'], 0., 'saved closure', 1e-9)
    close(s['complementarity'], 0., 'saved complementarity')
    Y = sum(y); F = dot(p, y); C = dot(e['c'], y); P = q*work
    baselines = game['selected']['br']
    if not isinstance(baselines, list) or len(baselines) != 3:
        raise ValueError('Incomplete saved platform-profit baselines')
    for b in baselines:
        required(b, ('platform', 'baseline'))
        if isinstance(b['platform'], bool) or b['platform'] not in (0, 1, 2):
            raise ValueError('Unknown platform-profit baseline')
    if {b['platform'] for b in baselines} != {0, 1, 2}:
        raise ValueError('Duplicate platform-profit baseline')
    for b in baselines:
        j = b['platform']
        close((p[j]-e['c'][j]-q*e['a'][j])*y[j], b['baseline'], 'saved platform profit')
    A = v*wait*Y
    rent = A if institution == 'transferable_payment' else 0.
    physical = wait if institution == 'real_waiting' else 0.
    G = c['congestion_cost']*(.15*rho*rho+max(rho-.9, 0.)**3)
    profit = F-C-P
    ts = cs+F-C-G+rent
    return dict(case=c['name'], institution=institution, charge=q, kappa=0.,
        prices=p, demand=y, utilization=rho, capacity=e['K'], potential_requests=sum(e['D']),
        CS=cs, TS=ts, air_passengers=Y, resource_work=work,
        physical_wait_minutes=physical, passenger_wait_minutes=physical*Y,
        equivalent_access_minutes=wait, value_per_access_minute=v,
        real_access_cost=A-rent, fare_payment=F, access_payment=rent,
        traveler_payment=F+rent, platform_profit=profit,
        public_charge_revenue=P, allocator_rent_revenue=rent,
        operating_resource_cost=C, external_cost=G,
        ts_identity_error=ts-(cs+profit+P+rent-G),
        transfer_cancellation_error=-(F+rent)+profit+P+rent+C)


def select_policy(menu):
    if not menu:
        raise ValueError('Empty policy menu')
    for r in menu:
        number(r['TS']); number(r['charge'])
    return max(menu, key=lambda r: (r['TS'], -r['charge']))


def aggregate(rows):
    return {key: dict(average=sum(r[key] for r in rows)/len(rows),
                     total=sum(r[key] for r in rows),
                     minimum=min(r[key] for r in rows), maximum=max(r[key] for r in rows))
            for key in METRICS}


def matched_comparison(raw):
    required(raw, ('records', 'games'))
    games = [g for g in raw['games'] if g['mode'] == 'equal_vot']
    keys = [(tuple(vector(g['profile'])), number(g['charge'])) for g in games]
    expected = {(p, q) for p in PROFILES for q in MENU}
    if len(keys) != 33 or set(keys) != expected:
        raise ValueError('Incomplete or duplicate matched 33-response menu')
    records = []
    for i, r in enumerate(raw['records']):
        required(r, ('dataclass', 'effective'))
        c = r['dataclass']
        required(c, ('demand_scale', 'capacity_scale', 'reliability_power'))
        profile = tuple(c[k] for k in ('demand_scale', 'capacity_scale', 'reliability_power'))
        if profile in PROFILES:
            records.append((i, r, profile))
    rk = [(p, r['dataclass']['congestion_cost']) for _, r, p in records]
    if len(rk) != 15 or set(rk) != {(p, E) for p in PROFILES for E in ES}:
        raise ValueError('Incomplete/duplicate matched 15 states')
    if len({r['dataclass']['name'] for _, r, _ in records}) != 15:
        raise ValueError('Duplicate state name')
    by_game = {(tuple(g['profile']), g['charge']): (raw['games'].index(g), g) for g in games}
    ledgers, selected, summaries, paired_errors = [], [], [], []
    for i, record, profile in sorted(records, key=lambda r: r[1]['dataclass']['name']):
        institution_menus = {}
        for inst in INSTITUTIONS:
            menu = []
            for q in MENU:
                gi, g = by_game[profile, q]
                row = reconstruct(record, g, inst)
                row['source'] = dict(file=ACCESS, record_pointer=f'/records/{i}', game_pointer=f'/games/{gi}')
                menu.append(row)
            institution_menus[inst] = menu
            ledgers.extend(menu)
            nc, sd = menu[0], select_policy(menu)
            selected.extend([dict(nc, policy='NC', gain_from_NC=0.),
                             dict(sd, policy='SD', gain_from_NC=sd['TS']-nc['TS'])])
        for a, b in zip(*(institution_menus[inst] for inst in INSTITUTIONS)):
            paired_errors.append(abs(b['TS']-a['TS']-a['real_access_cost']))
    for inst in INSTITUTIONS:
        for policy in ('NC', 'SD'):
            rows = [r for r in selected if r['institution'] == inst and r['policy'] == policy]
            summaries.append(dict(institution=inst, policy=policy, states=15,
                metrics=aggregate(rows), average_gain=sum(r['gain_from_NC'] for r in rows)/15,
                active_states=sum(r['charge'] > 0 for r in rows),
                selected_charge_histogram=histogram(rows)))
    gains = {r['institution']: r['average_gain'] for r in summaries if r['policy'] == 'SD'}
    real, rent = (gains[inst] for inst in INSTITUTIONS)
    reduction = 100*(1-rent/real)
    return dict(states=15, behavioral_profiles=3, saved_responses=33, menu=MENU,
        same_menu=True, kappa=0., selection='max TS; exact ties choose smaller charge',
        summaries=summaries, selected=selected, menu_ledgers=ledgers,
        claims=dict(real_waiting_gain=real, transferable_payment_gain=rent,
                    reduction_percent=reduction, rounded=[f'{x:.2f}' for x in (real, rent, reduction)]),
        max_transfer_cancellation_error=max(abs(r['transfer_cancellation_error']) for r in ledgers),
        max_ts_identity_error=max(abs(r['ts_identity_error']) for r in ledgers),
        max_common_charge_rent_identity_error=max(paired_errors))


def histogram(rows):
    return {f'{q:g}': n for q, n in sorted(Counter(r['charge'] for r in rows).items())}


def queue_record(c):
    required(c, ('base_demand', 'demand_scale', 'capacity_scale', 'base_capacity',
        'platform_affinity', 'differentiation', 'outside_affinity', 'price_sensitivity',
        'time_sensitivity', 'reliability_sensitivity', 'base_cost', 'cost_gap', 'exposure',
        'base_loss', 'platform_resource', 'service_time', 'logit_scale', 'outside_price',
        'outside_time', 'price_lower', 'price_upper'))
    e = dict(D=[x*c['demand_scale'] for x in vector(c['base_demand'])],
        alpha=[[v+(c['differentiation'] if g == j else 0.) for j, v in enumerate(row)]
               for g, row in enumerate(c['platform_affinity'])],
        alpha_outside=c['outside_affinity'], beta=c['time_sensitivity'],
        gamma=c['reliability_sensitivity'], mu=c['logit_scale'],
        c=[x+(0. if j == 0 else c['cost_gap'] if j == 1 else .5*c['cost_gap'])
           for j, x in enumerate(vector(c['base_cost']))],
        chi=c['exposure'], ell=c['base_loss'], a=c['platform_resource'],
        K=c['base_capacity']*c['capacity_scale'], service_time=c['service_time'],
        price_domain=[c['price_lower'], c['price_upper']],
        outside_price=c['outside_price'], outside_time=c['outside_time'])
    e['lambda'] = c['price_sensitivity']
    record = dict(dataclass=c, effective=e)
    effective(record)
    return record


def validate_queue_state(c, g):
    required(g, ('accepted', 'profile', 'cv2', 'charge', 'prices', 'demand', 'rho',
        'wait_minutes', 'cs', 'ts', 'profits', 'fare_receipts', 'resource_receipts',
        'real_cost', 'system_cost', 'ground_demand', 'potential_requests',
        'outside_price', 'probabilities', 'queue_cost', 'vot_mass'))
    if g['accepted'] is not True or g['profile'] != c['name']:
        raise ValueError('Queue response not accepted or wrong case')
    record = queue_record(c); e = record['effective']
    p, rho, wait = vector(g['prices']), number(g['rho']), number(g['wait_minutes'])
    q, cv2 = number(g['charge']), number(g['cv2'])
    if not 0 <= rho < 1 or wait < 0 or cv2 not in (0., 1., 2.) or q < 0:
        raise ValueError('Invalid saved stationary queue state')
    if min(p) < c['price_lower']-1e-9 or max(p) > c['price_upper']+1e-9:
        raise ValueError('Queue fare outside box')
    y, prob, cs = choice(e, c['reliability_power'], p, rho, wait, e['beta'])
    work = dot(e['a'], y); C = dot(e['c'], y); F = dot(p, y)
    G = c['congestion_cost']*(.15*rho*rho+max(rho-.9, 0.)**3)
    close(work/e['K'], rho, 'queue resource closure', 1e-9)
    wcalc = (1+cv2)*120*sum(a*a*z for a, z in zip(e['a'], y))/(2*e['K']**2*(1-rho))
    close(wcalc, wait, 'stationary queue formula')
    vot = sum(e['D'][i]*sum(prob[i])*e['beta'][i]/e['lambda'][i] for i in range(3))
    comparisons = dict(demand=y, probabilities=prob, cs=cs, ts=cs+F-C-G,
        profits=[(p[j]-e['c'][j]-q*e['a'][j])*y[j] for j in range(3)],
        fare_receipts=F, resource_receipts=q*work, real_cost=C, system_cost=G,
        ground_demand=sum(e['D'])-sum(y), potential_requests=sum(e['D']),
        outside_price=e['outside_price'], queue_cost=wait*vot, vot_mass=vot)
    for k, v in comparisons.items():
        close(v, g[k], 'queue '+k)


def queue_appraisals(games, profiles, cv_values, charges, scope, file):
    rows = []
    for profile in profiles:
        for cv in cv_values:
            menu = [g for g in games if g['profile'] == profile and g['cv2'] == cv and g['charge'] in charges]
            if sorted(g['charge'] for g in menu) != charges:
                raise ValueError('Incomplete or duplicate queue menu')
            for E in ES:
                for k in KAPPAS:
                    valued = []
                    for g in menu:
                        rho = g['rho']; shape = .15*rho*rho+max(rho-.9, 0.)**3
                        ts = g['ts']+(2800.-E)*shape+k*g['outside_price']*g['ground_demand']
                        valued.append(dict(charge=g['charge'], TS=ts, game=g))
                    opt = select_policy(valued); g = opt['game']
                    nc = next(r for r in valued if r['charge'] == 0)
                    rows.append(dict(scope=scope, profile=profile, cv2=cv, externality=E, kappa=k,
                        charge=opt['charge'], NC_TS=nc['TS'], SD_TS=opt['TS'],
                        gain=opt['TS']-nc['TS'], menu=charges, prices=g['prices'],
                        current_upper=opt['charge'] == max(charges), former_upper=opt['charge'] == 20,
                        fare_upper=any(abs(p-44.) <= 1e-7 for p in g['prices']),
                        source=dict(file=file,
                            collection=('games' if file == QUEUE else 'new_games' if opt['charge'] > 20 else 'old_games'),
                            selector=dict(profile=profile, cv2=cv, charge=opt['charge']))))
    return rows


def endpoint_summary(rows, menu, profiles):
    return dict(appraisals=len(rows), profiles=profiles, menu=menu,
        current_upper_charge=max(menu), current_upper_selections=sum(r['current_upper'] for r in rows),
        former_upper_charge=20, former_upper_selections=sum(r['former_upper'] for r in rows),
        above_former_upper_selections=sum(r['charge'] > 20 for r in rows),
        selected_charge_histogram=histogram(rows), fare_upper_charge=44,
        fare_upper_event_rows=sum(r['fare_upper'] for r in rows), fare_upper_denominator=len(rows))


def queue_comparison(base, extension):
    required(base, ('cases', 'games', 'design'))
    required(extension, ('cases', 'old_games', 'new_games', 'design'))
    if extension['cases'] != base['cases'] or extension['old_games'] != base['games']:
        raise ValueError('Expanded queue changed old cases or responses')
    cases = {c['name']: c for c in base['cases']}
    if len(cases) != 3 or len(base['cases']) != 3:
        raise ValueError('Expected three queue profiles')
    expected_profiles = {(1.35, .9, 1.7), (1.7, .7, 1.7), (1.12, .83, 1.93)}
    if {tuple(c[k] for k in ('demand_scale', 'capacity_scale', 'reliability_power')) for c in cases.values()} != expected_profiles:
        raise ValueError('Unrecognized queue profiles')
    if (base['design']['charges'] != COMMON or base['design']['fine_charges'] != [2, 6, 10, 14, 18]
        or base['design']['externalities'] != list(ES) or base['design']['kappas'] != list(KAPPAS)
        or base['design']['cv2'] != [0, 1, 2] or base['design']['horizon_minutes'] != 120
        or base['design']['behavior_externality'] != 2800):
        raise ValueError('Unrecognized historical queue menu/appraisal design')
    d = extension['design']
    if (d['profile'] != EXPANDED_PROFILE or d['baseline_common'] != COMMON
        or d['fixed_schedule'] != [[v, q] for q in (24, 28) for v in (0, 1, 2)]
        or d['externalities'] != list(ES) or d['kappas'] != list(KAPPAS)):
        raise ValueError('Unrecognized expanded menu')
    key = lambda g: (g['profile'], number(g['cv2']), number(g['charge']))
    expected = {(p, v, q) for p in cases for v in (0, 1, 2) for q in (MENU if v == 1 else COMMON)}
    found = [key(g) for g in base['games']]
    if len(found) != 69 or set(found) != expected:
        raise ValueError('Incomplete/duplicate historical queue responses')
    extra = [key(g) for g in extension['new_games']]
    if len(extra) != 6 or set(extra) != {(EXPANDED_PROFILE, v, q) for v in (0, 1, 2) for q in (24, 28)}:
        raise ValueError('Incomplete/duplicate expanded queue responses')
    for g in base['games']+extension['new_games']:
        validate_queue_state(cases[g['profile']], g)
    all_games = base['games']+extension['new_games']
    names = sorted(cases)
    definitions = (
        ('historical_common', base['games'], names, [0, 1, 2], COMMON, QUEUE),
        ('historical_fine', base['games'], names, [1], MENU, QUEUE),
        ('expanded_common', all_games, [EXPANDED_PROFILE], [0, 1, 2], COMMON+[24, 28], EXTENSION),
        ('expanded_fine', all_games, [EXPANDED_PROFILE], [1], MENU+[24, 28], EXTENSION),
    )
    result, rows = {}, []
    for label, games, profiles, cvs, charges, file in definitions:
        current = queue_appraisals(games, profiles, cvs, charges, label, file)
        rows.extend(current)
        result[label] = endpoint_summary(current, charges, profiles)
    result['policies'] = rows
    result['menu_warning'] = 'Extension covers one profile only; do not pool 20 and 28 as a single common upper bound.'
    result['fare_warning'] = 'Fare-upper events use p=44, not charge q=20/28; denominators are appraisal rows, not unique games.'
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def invalid(s):
        raise ValueError('Nonfinite JSON constant: '+s)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=True, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def validate_references(root, matched, queue):
    historical = [r for r in csv_rows(root/REFERENCES[0]) if r['model'] == 'equal_vot'
                  and float(r['kappa']) == 0 and float(r['real_share']) in (0, 1)]
    if len(historical) != 30:
        raise ValueError('Historical policy targets incomplete')
    index = {(r['case'], r['institution']): r for r in matched['selected'] if r['policy'] == 'SD'}
    for old in historical:
        inst = 'real_waiting' if float(old['real_share']) == 1 else 'transferable_payment'
        row = index[old['case'], inst]
        close(row['charge'], float(old['charge']), 'historical access selection')
        close(row['gain_from_NC'], float(old['gain']), 'historical access gain')
        close(row['air_passengers'], float(old['SD_served']), 'historical access passengers')
    summary = read(root/REFERENCES[1])
    for inst, share in zip(INSTITUTIONS, (1., 0.)):
        old = next(r for r in summary['summaries'] if r['model'] == 'equal_vot'
                   and r['real_share'] == share and r['kappa'] == 0)
        now = next(r for r in matched['summaries'] if r['institution'] == inst and r['policy'] == 'SD')
        close(old['average_gain'], now['average_gain'], 'historical access aggregate')
    def policy_key(r):
        return r['profile'], float(r['cv2']), float(r['externality']), float(r['kappa']), r['menu_kind']
    legacy = {policy_key(r): r for r in csv_rows(root/REFERENCES[4])}
    kinds = dict(historical_common='common6', historical_fine='native_fine11',
                 expanded_common='expanded_common', expanded_fine='expanded_fine')
    for row in queue['policies']:
        key = row['profile'], row['cv2'], row['externality'], row['kappa'], kinds[row['scope']]
        if key not in legacy:
            raise ValueError('Historical queue policy target absent')
        for field in ('charge', 'NC_TS', 'SD_TS', 'gain'):
            close(row[field], float(legacy[key][field]), 'historical queue '+field)
    old_common = [r for r in csv_rows(root/REFERENCES[2]) if r['menu_kind'] == 'common6']
    if len(old_common) != 135:
        raise ValueError('Historical common appraisal denominator differs')
    diag = read(root/REFERENCES[3])['summary']; ext = read(root/REFERENCES[5])
    for label, prefix in [('historical_common', 'common'), ('historical_fine', 'fine')]:
        close(queue[label]['appraisals'], diag[prefix+'_appraisals'])
        close(queue[label]['current_upper_selections'], diag[prefix+'_upper_endpoints'])
    close(queue['expanded_common']['current_upper_selections'], ext['remaining_upper_endpoints'])
    close(sum(queue[k]['fare_upper_event_rows'] for k in ('expanded_common', 'expanded_fine')),
          ext['expanded_policy_fare_upper'], 'historical fare events across 60 appraisals')
    return dict(access_policy_rows=30, queue_policy_rows=len(queue['policies']),
                historical_common_rows=135, historical_summary_agreement=True)


def compute(root):
    matched = matched_comparison(read(root/ACCESS))
    queue = queue_comparison(read(root/QUEUE), read(root/EXTENSION))
    checks = validate_references(root, matched, queue)
    return dict(schema=1, new_Nash_solves=0, new_fixed_point_solves=0,
                matched15=matched, queue=queue, historical_validation=checks)


def source_mapping(result, hashes):
    return dict(input_sha256=hashes, frozen_design=DESIGN,
        matched_selection=dict(file=ACCESS, mode='equal_vot', profiles=[list(p) for p in PROFILES],
            externalities=list(ES), kappa=0, charges=MENU, responses=33, appraisal_states=15),
        row_mapping='Every matched menu/selected ledger includes exact record/game JSON pointers. Queue rows include exact response selectors.',
        formulas=dict(TS='CS + platform_profit + public_charge_revenue + allocator_rent_revenue - external_cost',
            resource_work='sum(a_m*y_m)', air_passengers='sum(y_m)',
            traveler_payment='sum(p_m*y_m) + access_payment',
            platform_profit='sum((p_m-c_m-q*a_m)*y_m)',
            public_charge_revenue='q*sum(a_m*y_m)',
            allocator_rent_revenue='0 (real waiting), v*equivalent_access_minutes*sum(y_m) (payment)',
            real_access_cost='v*physical_wait_minutes*sum(y_m)',
            operating_resource_cost='sum(c_m*y_m)',
            external_cost='E*(0.15*rho^2+max(rho-0.90,0)^3)',
            transfer_cancellation='-traveler_payment+platform_profit+public_charge_revenue+allocator_rent_revenue+operating_resource_cost = 0',
            common_charge_identity='TS_payment-TS_real_waiting = real_access_cost_waiting',
            physical_wait='common minutes per air passenger; payment has zero physical waiting',
            CS='historical absolute logsum normalization, including access disutility',
            queue_current_upper='selected q == max(actual complete menu)',
            queue_former_upper='selected q == 20, including expanded menus with maximum 28',
            fare_upper='at least one selected platform fare within 1e-7 of 44'),
        interpretation='Matched designed economies and conditional institutional accounting; not observed markets or a proof that physical waiting is unavoidable.',
        queue_menu_warning=result['queue']['menu_warning'])


def table_data(result):
    summaries = result['matched15']['summaries']
    labels = dict(TS='Absolute TS', CS='Consumer surplus', air_passengers='Air passengers',
        resource_work='Resource work', physical_wait_minutes='Common physical wait (min)',
        passenger_wait_minutes='Passenger waiting (min)', real_access_cost='Real access cost',
        fare_payment='Traveler fares', access_payment='Traveler access payment',
        traveler_payment='Traveler monetary payment', platform_profit='Platform profit',
        public_charge_revenue='Public charge revenue', allocator_rent_revenue='Allocation rent revenue',
        operating_resource_cost='Operating resource cost', external_cost='System external cost',
        equivalent_access_minutes='Equivalent access expense (min)')
    lines = ['% Columns: metric, real waiting NC, real waiting SD, payment NC, payment SD.',
             '% Arithmetic averages over the same 15 states; kappa=0; monetary values in model dollars.']
    for metric in METRICS:
        values = [r['metrics'][metric]['average'] for r in summaries]
        lines.append(labels[metric]+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines.append('TS gain from own NC & '+' & '.join(f"{r['average_gain']:.2f}" for r in summaries)+r' \\')
    return '\n'.join(lines)+'\n'


def queue_table_data(result):
    lines = ['% Scope & appraisals & current upper & at current upper & at former upper 20 & above 20 & fare-upper rows.']
    for label in ('historical_common', 'historical_fine', 'expanded_common', 'expanded_fine'):
        r = result['queue'][label]
        values = [r[k] for k in ('appraisals', 'current_upper_charge', 'current_upper_selections',
                                 'former_upper_selections', 'above_former_upper_selections', 'fare_upper_event_rows')]
        lines.append(label.replace('_', ' ')+' & '+' & '.join(str(v) for v in values)+r' \\')
    return '\n'.join(lines)+'\n'


def run(root, folder):
    root, folder = Path(root), Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    inputs = tuple(PINNED)+REFERENCES+(DESIGN,)
    hashes = {p: digest(root/p) for p in inputs}
    for p, expected in PINNED.items():
        if hashes[p] != expected:
            raise ValueError('Frozen source hash differs: '+p)
    result = compute(root)
    folder.mkdir(parents=True)
    for name in inputs:
        target = folder/'inputs'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    for name in OWNED:
        target = folder/'execution_source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    dump(folder/'results.json', result)
    dump(folder/'source_mapping.json', source_mapping(result, hashes))
    (folder/'table_data.tex').write_text(table_data(result), encoding='utf-8')
    (folder/'queue_table_data.tex').write_text(queue_table_data(result), encoding='utf-8')
    generated = ['results.json', 'source_mapping.json', 'table_data.tex', 'queue_table_data.tex']
    output_hashes = {name: digest(folder/name) for name in generated}
    execution = {name: digest(root/name) for name in OWNED}
    dump(folder/'manifest.json', dict(schema=1, inputs=hashes, outputs=output_hashes, execution=execution))
    if hashes != {p: digest(root/p) for p in inputs}:
        raise ValueError('Source changed during arithmetic generation')
    replay(folder)
    return result


def replay(folder):
    folder = Path(folder); manifest = read(folder/'manifest.json')
    for key, prefix in [('inputs', 'inputs'), ('outputs', ''), ('execution', 'execution_source')]:
        for name, expected in manifest[key].items():
            if digest(folder/prefix/name) != expected:
                raise ValueError('Replay hash mismatch: '+name)
    for p, expected in PINNED.items():
        if manifest['inputs'][p] != expected:
            raise ValueError('Replay changed frozen input hash')
    result = compute(folder/'inputs')
    if result != read(folder/'results.json'):
        raise ValueError('Replay result differs')
    if source_mapping(result, manifest['inputs']) != read(folder/'source_mapping.json'):
        raise ValueError('Replay source mapping differs')
    for name, content in [('table_data.tex', table_data(result)), ('queue_table_data.tex', queue_table_data(result))]:
        if (folder/name).read_text(encoding='utf-8') != content:
            raise ValueError('Replay table differs: '+name)
    return result
