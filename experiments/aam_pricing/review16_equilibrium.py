"""Full-domain deviation bounds for exact manuscript response vectors.

Payment access changes waiting disutility, not fixed service or outside utility.
The interval operations and global partition method are inherited unchanged.
"""
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from .review15_bounds import Problem, I, hull


class AccessProblem(Problem):
    def __init__(self, case, excluded=(), waiting_coefficients=None):
        super().__init__(case, excluded)
        self.waiting_coefficients = list(map(I, waiting_coefficients or case['time_sensitivity']))
        if min(x.lo for x in self.waiting_coefficients) <= 0:
            raise ValueError('Positive waiting disutility required')

    def evaluate(self, base, wait):
        probabilities, thetas, work = [], [], I(0)
        for g, (total, shares, average) in enumerate(base):
            odds = total * (-self.waiting_coefficients[g] * wait / self.mu).exp()
            theta = I((I(odds.lo) / (1 + I(odds.lo))).lo,
                      (I(odds.hi) / (1 + I(odds.hi))).hi)
            theta = I(max(Decimal(0), theta.lo), min(Decimal(1), theta.hi))
            probabilities.append([theta * s for s in shares])
            thetas.append(theta)
            work += self.D[g] * theta * average
        y = [sum((self.D[g] * probabilities[g][j] for g in range(3)), I(0)) for j in range(3)]
        return dict(prob=probabilities, theta=thetas, demand=y, work=work, base=base)

    def gradient(self, p, q, j):
        gradients = []
        for s in self.enclose(p):
            prob, theta = s['prob'], s['theta']
            direct = -sum((self.D[g] * self.lam[g] / self.mu * prob[g][j] * (1-prob[g][j])
                           for g in range(3)), I(0))
            work_price = -sum((self.D[g] * self.lam[g] / self.mu * prob[g][j]
                               * (self.a[j]-sum((self.a[k]*prob[g][k] for k in self.active), I(0)))
                               for g in range(3)), I(0))
            if s['regime'] == 'binding':
                wait_j = -sum((self.D[g]*self.waiting_coefficients[g]/self.mu*prob[g][j]*(1-theta[g])
                               for g in range(3)), I(0))
                wait_R = -sum((self.D[g]*self.waiting_coefficients[g]/self.mu*theta[g]*(1-theta[g])
                               * s['base'][g][2] for g in range(3)), I(0))
                dy = direct-wait_j*work_price/wait_R
            else:
                partial = [-self.hp(s['rho'])*sum((self.D[g]*self.gamma[g]/self.mu*prob[g][k]
                           * (self.chi[k]-sum((self.chi[h]*prob[g][h] for h in self.active), I(0)))
                           for g in range(3)), I(0)) for k in range(3)]
                Rrho = sum((self.a[k]*partial[k] for k in self.active), I(0))
                Rrho = I(Rrho.lo, min(Rrho.hi, (self.K*self.closure_slope_upper).hi))
                dy = direct+partial[j]*work_price/(self.K-Rrho)
            gradients.append(s['demand'][j]+(p[j]-self.cost[j]-I(q)*self.a[j])*dy)
        return hull(gradients)

    def welfare(self, *args, **kwargs):
        if any(x.lo != y.lo or x.hi != y.hi for x, y in zip(self.beta, self.waiting_coefficients)):
            raise NotImplementedError('Access welfare requires explicit monetary-transfer accounting')
        return super().welfare(*args, **kwargs)


def collect_inputs(root):
    root = Path(root)
    rows = []

    def append(identifier, case, candidate, source, roles, mode='native', profile=None):
        waiting = list(case['time_sensitivity'])
        if mode == 'equal_vot':
            D = [x*case['demand_scale'] for x in case['base_demand']]
            v = sum(d*b for d, b in zip(D, case['time_sensitivity'])) / sum(
                d*l for d, l in zip(D, case['price_sensitivity']))
            waiting = [l*v for l in case['price_sensitivity']]
        rows.append(dict(id=identifier, case=case, prices=candidate['prices'],
                         charge=float(candidate['charge']), excluded=candidate.get('excluded', []),
                         waiting_coefficients=waiting, source=source, roles=roles,
                         mode=mode, profile=profile))

    normalized_path = 'results/trb_review14_policy_v1/normalized_menus.json'
    normalized = json.loads((root/normalized_path).read_text(encoding='utf-8'))
    by_id = {x['id']: x for x in normalized}
    path = 'results/trb_review10_core_v1/paths.csv'
    with (root/path).open(encoding='utf-8-sig', newline='') as stream:
        for x in csv.DictReader(stream):
            if x['menu'] not in ('joint_2_c0', 'resource_2_c1', 'resource_3_c1'):
                continue
            q = float(x['charge'])
            append(f"figure2:{x['menu']}:q{q:g}", by_id[x['menu']]['case'],
                   dict(prices=json.loads(x['prices']), charge=q, excluded=[]),
                   dict(file=path, menu=x['menu'], charge=q), ['figure2'])
    for menu in normalized:
        for x in menu['candidates']:
            append(f"supplement:{menu['id']}:q{x['charge']:g}", menu['case'], x,
                   dict(file=normalized_path, menu=menu['id'], charge=x['charge']), ['appendix_menu'])

    path = 'results/trb_review12_signals_v1/normalized_cases.json'
    test = json.loads((root/path).read_text(encoding='utf-8'))['test']
    for menu in test:
        if menu['case']['congestion_cost'] != 90:
            continue
        for x in menu['candidates']:
            append(f"policy:{menu['case_id']}:q{x['charge']:g}", menu['case'], x,
                   dict(file=path, menu=menu['id'], charge=x['charge']), ['figures7_8', 'policy'],
                   profile=menu.get('behavior_profile'))

    path = 'results/trb_review6_access_v1/saved_inputs.json'
    access = json.loads((root/path).read_text(encoding='utf-8'))
    profiles = {(.85, 1.12, 2.2), (1.35, .9, 2.2), (1.7, .7, 2.2)}
    for game in access['games']:
        profile = tuple(game['profile'])
        if profile not in profiles:
            continue
        case = next(x['dataclass'] for x in access['records']
                    if x['dataclass']['congestion_cost'] == 90
                    and (x['dataclass']['demand_scale'], x['dataclass']['capacity_scale'],
                         x['dataclass']['reliability_power']) == profile)
        append('access:'+game['game'], case,
               dict(prices=game['selected']['prices'], charge=game['charge']),
               dict(file=path, game=game['game']), ['figure4'], game['mode'], list(profile))
    if len({x['id'] for x in rows}) != len(rows):
        raise ValueError('Input identity collision')
    return rows


def behavior_key(row):
    case = {k: v for k, v in row['case'].items() if k not in ('name', 'congestion_cost')}
    payload = [case, row['prices'], row['charge'], row['excluded'], row['waiting_coefficients']]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
