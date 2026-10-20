"""Task A: bounded kink repair and outward-rounded whole-fare envelopes.

Historical engines are imported read-only. Numerical search acceptance and
entire-domain profit upper bounds are deliberately different result fields.
"""
from __future__ import annotations

from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
import hashlib
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from .config import MarketCase
from .review90_waiting import WaitingEngine
from .review_equilibrium import ReviewOutcome, projected_gradient
from .review_hard_capacity_repair import EvaluationBudget, project_transition
from .review5_best_response import audit_candidate
from .review6_heterogeneity import snapshot
from .review11_participation_audit import read, jsonl, merge_attempts, HET
from .review11_optout import OptOutEngine

OUTPUT = 'results/trb_review13_equilibrium_v1'
QA = 'docs/qa/trb_thirteenth_review_20261018'
OWNED = ['experiments/aam_pricing/review13_equilibrium.py',
         'scripts/run_review13_equilibrium.py', 'tests/test_review13_equilibrium.py']
INPUTS = [HET+'/'+x for x in ('design.json', 'games.jsonl', 'rescue_games.jsonl')]
INPUTS += ['results/trb_review11_optout_v1/games.jsonl']
DOWN = Context(prec=40, rounding=ROUND_FLOOR)
UP = Context(prec=40, rounding=ROUND_CEILING)
NEAR = Context(prec=40, rounding=ROUND_HALF_EVEN)


def decimal(x):
    if isinstance(x, Decimal):
        return x
    return Decimal.from_float(float(x)) if isinstance(x, (float, np.floating)) else Decimal(x)


class Interval:
    """Exact binary inputs; directed basic ops; adjacent correctly-rounded exp/ln."""
    def __init__(self, lo, hi=None):
        self.lo, self.hi = decimal(lo), decimal(lo if hi is None else hi)
        if self.lo > self.hi:
            raise ValueError('Reversed interval')

    def __add__(self, other):
        b = interval(other)
        return Interval(DOWN.add(self.lo, b.lo), UP.add(self.hi, b.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -interval(other)

    def __rsub__(self, other):
        return interval(other) + -self

    def __mul__(self, other):
        b = interval(other)
        pairs = [(a, c) for a in (self.lo, self.hi) for c in (b.lo, b.hi)]
        return Interval(min(DOWN.multiply(a, c) for a, c in pairs),
                        max(UP.multiply(a, c) for a, c in pairs))

    __rmul__ = __mul__

    def __truediv__(self, other):
        b = interval(other)
        if b.lo <= 0 <= b.hi:
            raise ValueError('Division interval contains zero')
        return self * Interval(DOWN.divide(Decimal(1), b.hi), UP.divide(Decimal(1), b.lo))

    def exp(self):
        return Interval(NEAR.next_minus(NEAR.exp(self.lo)), NEAR.next_plus(NEAR.exp(self.hi)))

    def log(self):
        if self.lo <= 0:
            raise ValueError('Positive logarithm domain required')
        return Interval(NEAR.next_minus(NEAR.ln(self.lo)), NEAR.next_plus(NEAR.ln(self.hi)))

    def intersection(self, other):
        b = interval(other)
        lo, hi = max(self.lo, b.lo), min(self.hi, b.hi)
        return Interval(lo, hi) if lo <= hi else None

    def serialized(self):
        return [str(self.lo), str(self.hi)]


def interval(x):
    return x if isinstance(x, Interval) else Interval(x)


def midpoint(x):
    return NEAR.divide(NEAR.add(x.lo, x.hi), Decimal(2))


class Envelope:
    """All closure roots are covered without monotonicity in rho or own fare."""
    def __init__(self, engine):
        self.e = engine
        self.excluded = list(getattr(engine, 'excluded', []))
        self.active = [i for i in range(3) if i not in self.excluded]
        self.N = sum((Interval(x) for x in engine.demand_base), Interval(0))
        self.K = Interval(engine.capacity)
        self.a = [Interval(x) for x in engine.resource]
        self.c = [Interval(x) for x in engine.cost]
        self.mu = Interval(engine.logit_scale)
        self.outside = [(Interval(engine.outside_affinity[g]) -
                         Interval(engine.price_sensitivity[g])*engine.case.outside_price -
                         Interval(engine.time_sensitivity[g])*engine.case.outside_time)/self.mu
                        for g in range(3)]

    def h(self, rho):
        q = interval(rho)
        nu = Interval(self.e.case.reliability_power)
        if q.lo == q.hi == 0:
            power = Interval(0)
        elif q.lo == 0:
            power = Interval(0, (Interval(q.hi).log()*nu).exp().hi)
        else:
            power = (q.log()*nu).exp()
        excess = Interval(max(Decimal(0), (q-Interval(.88)).lo),
                          max(Decimal(0), (q-Interval(.88)).hi))
        return Interval(.55)*power + Interval(1.10)*excess*excess

    def choice(self, prices, rho, wait):
        e = self.e
        h = self.h(rho)
        demand = [Interval(0) for _ in range(3)]
        cs = Interval(0)
        for g in range(3):
            weights = []
            for i in self.active:
                loss = Interval(e.base_loss[i])+Interval(e.exposure[i])*h
                u = (Interval(e.platform_affinity[g, i]) -
                     Interval(e.price_sensitivity[g])*prices[i] -
                     Interval(e.time_sensitivity[g])*e.time[i] -
                     Interval(e.reliability_sensitivity[g])*loss -
                     Interval(e.beta[g])*wait)/self.mu - self.outside[g]
                weights.append(u.exp())
            den = 1 + sum(weights, Interval(0))
            mass = Interval(e.demand_base[g])
            cs += mass*self.mu/Interval(e.price_sensitivity[g])*(self.outside[g]+den.log())
            for j, i in enumerate(self.active):
                # A softmax share increases with its own weight and decreases
                # with every competing weight, including the fixed outside 1.
                lowden = Interval(weights[j].lo) + 1 + sum(
                    (Interval(w.hi) for k, w in enumerate(weights) if k != j), Interval(0))
                highden = Interval(weights[j].hi) + 1 + sum(
                    (Interval(w.lo) for k, w in enumerate(weights) if k != j), Interval(0))
                share = Interval((Interval(weights[j].lo)/lowden).lo,
                                 (Interval(weights[j].hi)/highden).hi)
                demand[i] += mass*share
        work = sum((a*y for a, y in zip(self.a, demand)), Interval(0))
        return dict(demand=demand, work=work, cs=cs)

    def closures(self, prices):
        result = []
        rho = Interval(0, 1)
        # Fixed-point interval contraction preserves every possible free root.
        for _ in range(24):
            free = self.choice(prices, rho, Interval(0))
            narrowed = rho.intersection(free['work']/self.K)
            if narrowed is None:
                break
            if narrowed.lo == rho.lo and narrowed.hi == rho.hi:
                rho = narrowed
                break
            rho = narrowed
        else:
            free = self.choice(prices, rho, Interval(0))
        if narrowed is not None:
            free = self.choice(prices, rho, Interval(0))
            result.append(dict(free, regime='free', rho=rho, wait=Interval(0)))
        at_zero = self.choice(prices, Interval(1), Interval(0))
        if at_zero['work'].hi >= self.K.lo:
            upper = Decimal(1)
            for _ in range(32):
                if self.choice(prices, Interval(1), Interval(upper))['work'].hi < self.K.lo:
                    break
                upper = UP.multiply(upper, Decimal(2))
            else:
                raise ValueError('Failed to enclose the entire nonnegative wait interval')
            # For fixed fares, group waiting scales every aerial weight equally.
            # Its resource work is positive conditional mean work times aerial
            # probability, hence strictly decreasing, even for unequal a_m.
            ll, lh, ul, uh = Decimal(0), upper, Decimal(0), upper
            steps = 48 if all(p.lo == p.hi for p in prices) else 16
            for _ in range(steps):
                lowmid = NEAR.divide(NEAR.add(ll, lh), Decimal(2))
                work = self.choice(prices, Interval(1), Interval(lowmid))['work']
                if work.lo > self.K.hi:
                    ll = lowmid
                else:
                    lh = lowmid
                upmid = NEAR.divide(NEAR.add(ul, uh), Decimal(2))
                work = self.choice(prices, Interval(1), Interval(upmid))['work']
                if work.hi < self.K.lo:
                    uh = upmid
                else:
                    ul = upmid
            wait = Interval(ll, uh)
            binding = self.choice(prices, Interval(1), wait)
            result.append(dict(binding, regime='binding', rho=Interval(1), wait=wait))
        return result

    def profit_box(self, prices, charge, platform):
        closures = self.closures(prices)
        if not closures:
            return None
        values = [(prices[platform]-self.c[platform]-Interval(charge)*self.a[platform]) *
                  c['demand'][platform] for c in closures]
        return Interval(min(x.lo for x in values), max(x.hi for x in values))

    def unilateral(self, prices, charge, platform, max_leaves=32, seconds=2.):
        start = perf_counter()
        p = [Interval(x) for x in prices]
        baseline = self.profit_box(p, charge, platform)
        if baseline is None:
            raise ValueError('Baseline closure enclosure empty')
        domain = Interval(self.e.case.price_lower, self.e.case.price_upper)

        def bound(fares):
            q = p.copy()
            q[platform] = fares
            return self.profit_box(q, charge, platform)

        first = bound(domain)
        if first is None:
            raise ValueError('Whole fare interval has no closure enclosure')
        leaves = [(domain, first)]
        excluded = []
        while len(leaves)+len(excluded) < max_leaves and perf_counter()-start < seconds:
            index = max(range(len(leaves)), key=lambda i: leaves[i][1].hi)
            old_domain, old_value = leaves[index]
            cut = midpoint(old_domain)
            if cut in (old_domain.lo, old_domain.hi):
                break
            children = [Interval(old_domain.lo, cut), Interval(cut, old_domain.hi)]
            # Keep the parent until BOTH children have valid bounds. An exception
            # or a stop can therefore never silently erase uncovered prices.
            child_values = [bound(child) for child in children]
            leaves.pop(index)
            for child, value in zip(children, child_values):
                if value is None:
                    excluded.append(child.serialized())
                else:
                    leaves.append((child, value))
            if not leaves:
                raise ValueError('All fare enclosures empty at a feasible baseline')
        upper = max(v.hi for _, v in leaves)
        gap = max(Decimal(0), UP.subtract(upper, baseline.lo))
        return dict(platform=platform+1, fare_domain=[float(domain.lo), float(domain.hi)],
                    profit_upper=str(upper), baseline_lower=str(baseline.lo),
                    baseline_upper=str(baseline.hi), gain_upper=str(gap),
                    coverage_complete=True, epsilon_nash_closed=gap <= Decimal('1e-6'),
                    arithmetic='Decimal40 directed; adjacent correctly-rounded exp/ln',
                    terminal_intervals=len(leaves), empty_closure_intervals=excluded,
                    intervals=[dict(fare=x.serialized(), profit=v.serialized()) for x, v in leaves],
                    elapsed_seconds=perf_counter()-start)

    def welfare_box(self, kappa):
        e = self.e
        p = [Interval(e.case.price_lower, e.case.price_upper) for _ in range(3)]
        csmax = self.choice([Interval(e.case.price_lower) for _ in range(3)], Interval(0), Interval(0))['cs']
        csmin = sum((Interval(e.demand_base[g])*self.mu/Interval(e.price_sensitivity[g])*self.outside[g]
                     for g in range(3)), Interval(0))
        lower = csmin
        for i in self.active:
            negative = min(Decimal(0), (p[i]-self.c[i]).lo)
            ymax = min(self.N.hi, (self.K/self.a[i]).hi)
            lower += Interval(negative)*Interval(ymax)
        margins = [max(Decimal(0), (p[i]-self.c[i]).hi) for i in self.active]
        maxrate = max((Interval(x)/self.a[i]).hi for x, i in zip(margins, self.active))
        gains = min((self.K*Interval(maxrate)).hi, (self.N*Interval(max(margins))).hi)
        # G is nonnegative and maximal at rho=1. Ground net income is positive:
        # W^(TS,kappa) = TS + kappa*p0*y0, with 0 <= y0 <= N.
        gmax = Interval(e.case.congestion_cost)*(Interval(.15)+Interval(.1)*Interval(.1)*Interval(.1))
        lower -= gmax
        upper = csmax + Interval(gains) + Interval(kappa)*Interval(e.case.outside_price)*self.N
        return dict(lower=str(lower.lo), upper=str(upper.hi), kappa=kappa,
                    scope='Any capacity-feasible fare/wait response, not just Nash',
                    arithmetic='Exact effective binary primitives; Decimal40 outward bounds')


def orient_sides(work_price, binding, free):
    d = np.asarray(work_price)
    return np.where(d <= 0, binding, free), np.where(d <= 0, free, binding)


def kink_data(engine, prices, charge):
    p = np.asarray(prices, float)
    y, prob, cs = engine.choice(p, 1., 0.)
    weights = engine.demand_base*engine.price_sensitivity/engine.logit_scale
    direct = np.einsum('g,gm,gj->mj', weights, prob, prob)-np.diag(weights@prob)
    work_price = engine.resource@direct
    binding = engine.gradient(p, charge, ReviewOutcome(y, 1., prob, cs, True, 0., 0.))
    free = engine.gradient(p, charge, ReviewOutcome(y, 1., prob, cs, False, 0., 0.))
    left, right = orient_sides(work_price, binding, free)
    return dict(gap=float(engine.resource@y/engine.capacity-1),
                work_price=work_price, left=left, right=right,
                binding=binding, free=free)


def kink_residual(engine, prices, charge):
    d = kink_data(engine, prices, charge)
    p = np.asarray(prices)
    lv = np.where(p > engine.case.price_lower+1e-8, np.minimum(d['left'], 0), 0)
    rv = np.where(p < engine.case.price_upper-1e-8, np.maximum(d['right'], 0), 0)
    return np.r_[100*d['gap'], lv, rv]


def make_engine(menu):
    return WaitingEngine(MarketCase(**menu['case']))


def inventory(root):
    root = Path(root)
    spec = read(root/HET/'design.json')
    original = jsonl(root/HET/'games.jsonl')
    rescue = jsonl(root/HET/'rescue_games.jsonl')
    merged = merge_attempts(original, rescue)
    if len(merged) != 268 or sum(bool(g['selected']) for g in merged) != 263:
        raise ValueError('Frozen original/rescue coverage changed')
    return spec, merged, dict(original=original, rescue=rescue)


def frozen_design():
    return dict(schema=1, max_games=5, max_starts=3, game_seconds=85,
                hard_game_seconds=89, start_seconds=20, max_nfev=160,
                extend_on_failure=False, gain_tolerance=1e-6,
                max_interval_leaves=32, coordinate_interval_seconds=2., envelope_seconds=140,
                selection='Closest accepted candidate to nearest saved accepted charge; lexicographic tie',
                global_regularity_assumed=False, all_equilibria_enumerated=False)


def result_schema():
    return {
        'schema_version': 1,
        'games.jsonl': ['game', 'menu', 'charge', 'attempts', 'candidates', 'selected',
                        'numerical_accepted', 'errors', 'child_error', 'child_process_seconds'],
        'attempts.jsonl': ['game', 'event:start|finish', 'start', 'initial', 'fits',
                           'candidate', 'error', 'evaluations', 'choice_calls'],
        'candidate': ['prices', 'first_order_ok', 'transition_gap', 'oriented_kink_residual',
                      'work_price', 'left', 'right', 'old_direction_wrong_platforms', 'state',
                      'finite_audit', 'numerical_accepted'],
        'profit_bounds.jsonl': ['game', 'roles', 'excluded', 'prices', 'charge',
                                'coordinates', 'epsilon_nash_closed'],
        'profit_bounds.coordinate': ['platform', 'fare_domain', 'profit_upper', 'baseline_lower',
                                     'baseline_upper', 'gain_upper', 'coverage_complete',
                                     'epsilon_nash_closed', 'arithmetic', 'terminal_intervals',
                                     'intervals', 'empty_closure_intervals', 'elapsed_seconds'],
        'menus.json': ['menu', 'kappa', 'planned_charges', 'points', 'complete',
                      'possible_optimal_charges', 'optimum_welfare_interval', 'NC_interval',
                      'gain_over_NC_interval', 'selected_best_known_charge', 'selected_optimal_charge'],
        'menu.point': ['game', 'charge', 'known', 'source', 'candidate_TS', 'lower', 'upper',
                       'rigorous_unresolved_welfare', 'selected_TS'],
        'numeric_types': 'Bound endpoints: decimal strings. Numerical diagnostics: JSON numbers.',
        'limitations': ['all_equilibria_enumerated', 'no_nonexistence_inference',
                        'numerical_acceptance_is_not_whole_domain_epsilon',
                        'menu_completion_is_only_for_discovered_candidates',
                        'effective_binary_primitives_not_float_execution_roundoff'],
    }


def counterexample():
    fares = [5., 15., 35.]
    values = []
    for p in fares:
        values.append(float(p*(100*np.exp(5-p)/(1+np.exp(5-p)+np.exp(-44))+
                               10*np.exp(40-p)/(1+np.exp(40-p)+np.exp(-44)))))
    return dict(fares=fares, profits=values, interpolation_weight=(35-15)/(35-5),
                violates_quasiconcavity=bool(values[1] < min(values[0], values[2])),
                nonexistence_inference=False, structured_proposition_refuted=False)


def policy_interval(rows):
    rows = [dict(r, lower=decimal(r['lower']), upper=decimal(r['upper'])) for r in rows]
    floor = max(r['lower'] for r in rows)
    ceiling = max(r['upper'] for r in rows)
    nc = next(r for r in rows if r['charge'] == 0)
    alternatives = [r for r in rows if r['charge'] != 0]
    return dict(complete=all(r['known'] for r in rows),
                possible_optimal_charges=sorted(r['charge'] for r in rows if r['upper'] >= floor),
                optimum_welfare_interval=[str(floor), str(ceiling)],
                NC_interval=[str(nc['lower']), str(nc['upper'])],
                gain_over_NC_interval=[str(max(Decimal(0), DOWN.subtract(
                    max(r['lower'] for r in alternatives), nc['upper']))),
                    str(max(Decimal(0), UP.subtract(max(r['upper'] for r in alternatives), nc['lower'])))])


def saved_surplus(state, engine, kappa):
    """Outward arithmetic on saved numbers, not an enclosure of state-solver error."""
    if not 0 <= kappa <= 1 or engine.case.outside_price < 0:
        raise ValueError('Ground accounting requires kappa in [0,1] and p0 >= 0')
    ground = Interval(state['ground']) if 'ground' in state else Interval(state['potential'])-Interval(state['served'])
    return Interval(state['TS'])+Interval(kappa)*Interval(engine.case.outside_price)*ground


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')
        f.flush()


def freeze(root, folder):
    root, folder = Path(root), Path(folder)
    if folder.exists():
        raise FileExistsError('Task A output is immutable; no second execution')
    spec, merged, attempts = inventory(root)
    missing = [g for g in merged if not g['selected']]
    if len(missing) != 5:
        raise ValueError('Only the five frozen missing games are authorized')
    sources = OWNED + [str(p.relative_to(root)).replace('\\', '/')
                       for p in (root/'experiments/aam_pricing').glob('*.py')
                       if not p.name.startswith('review13_')]
    paths = sources + INPUTS + [QA+'/equilibrium_design.md',
                               'submission_packages/TRA_V290_LaTeX_Project/MAIN.tex']
    hashes = {p: digest(root/p) for p in paths}
    folder.mkdir(parents=True)
    for p in INPUTS + OWNED + [QA+'/equilibrium_design.md']:
        dst = folder/'frozen'/p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/p, dst)
    data = dict(spec=spec, games=merged, attempts=attempts,
                optout=jsonl(root/INPUTS[-1]), missing_ids=[g['game'] for g in missing])
    dump(folder/'inputs.json', data)
    dump(folder/'design.json', frozen_design())
    dump(folder/'schema.json', result_schema())
    dump(folder/'freeze.json', dict(source_sha256=hashes, inputs_sha256=digest(folder/'inputs.json'),
                                   design_sha256=digest(folder/'design.json'), frozen_before_solves=True))
    return dict(missing_ids=data['missing_ids'], design=frozen_design())


def verify_freeze(root, folder, include_main=True):
    saved = read(Path(folder)/'freeze.json')
    for p, expected in saved['source_sha256'].items():
        if not include_main and p.endswith('/MAIN.tex'):
            continue
        if digest(Path(root)/p) != expected:
            raise ValueError('Frozen dependency changed: '+p)
    for p in ('inputs', 'design'):
        if digest(Path(folder)/(p+'.json')) != saved[p+'_sha256']:
            raise ValueError('Frozen '+p+' changed')


def starts_for(data, gid):
    game = next(g for g in data['games'] if g['game'] == gid)
    records = [g for source in data['attempts'].values() for g in source if g['game'] == gid]
    candidates = [c for g in records for c in g.get('audited_candidates', [])]
    best = min(candidates, key=lambda c:c['max_gain'])['prices']
    neighbors = [g for g in data['games'] if g['menu'] == game['menu'] and g['selected']]
    neighbor = min(neighbors, key=lambda g:(abs(g['charge']-game['charge']), g['charge']))
    anchor = neighbor['selected']['prices']
    values, seen = [], set()
    for label, p in [('saved_least_gain', best), ('nearest_saved_charge', anchor), ('fixed22', [22.]*3)]:
        if tuple(p) not in seen:
            values.append((label, p))
            seen.add(tuple(p))
    return game, anchor, values


def solve_child(folder, gid):
    folder = Path(folder)
    data, plan = read(folder/'inputs.json'), read(folder/'design.json')
    if gid not in data['missing_ids'] or (folder/('game_'+gid.replace(':', '_')+'.json')).exists():
        raise ValueError('Only one execution of each of the five frozen IDs is allowed')
    game, anchor, starts = starts_for(data, gid)
    menu = next(m for m in data['spec']['menus'] if m['id'] == game['menu'])
    begin = perf_counter()
    engine = make_engine(menu)
    row = dict(game=gid, menu=game['menu'], charge=game['charge'], attempts=[],
               numerical_accepted=False, selected=None, candidates=[], errors=[])
    for label, initial in starts[:plan['max_starts']]:
        deadline = min(begin+plan['game_seconds'], perf_counter()+plan['start_seconds'])
        engine.budget = EvaluationBudget(limit=60000, deadline=deadline)
        attempt = dict(start=label, initial=initial, fits=[])
        row['attempts'].append(attempt)
        append(folder/'attempts.jsonl', dict(game=gid, event='start', **attempt))
        try:
            def residual(p):
                engine.budget.spend('oriented_kink_fit', p)
                return kink_residual(engine, p, game['charge'])
            fit = least_squares(residual, np.clip(initial, engine.case.price_lower+1e-8,
                                                  engine.case.price_upper-1e-8),
                                bounds=(engine.case.price_lower, engine.case.price_upper),
                                max_nfev=plan['max_nfev'], xtol=1e-12, ftol=1e-12, gtol=1e-12)
            p = fit.x
            attempt['fits'].append(dict(kind='oriented_transition', success=bool(fit.success),
                status=int(fit.status), nfev=int(fit.nfev), prices=p.tolist(),
                residual=residual(p).tolist(), message=str(fit.message)))
            projected = project_transition(engine, p)
            if projected:
                p = np.array(projected['prices'])
            kd = kink_data(engine, p, game['charge'])
            kr = float(np.max(abs(kink_residual(engine, p, game['charge']))))
            if kr > 1e-5:
                def smooth(p):
                    return projected_gradient(p, engine.gradient(p, game['charge']),
                                              engine.case.price_lower, engine.case.price_upper)
                fit2 = least_squares(smooth, np.clip(initial, engine.case.price_lower+1e-8,
                                                    engine.case.price_upper-1e-8),
                    bounds=(engine.case.price_lower, engine.case.price_upper),
                    max_nfev=plan['max_nfev'], xtol=1e-12, ftol=1e-12, gtol=1e-12)
                attempt['fits'].append(dict(kind='smooth_fallback', success=bool(fit2.success),
                    status=int(fit2.status), nfev=int(fit2.nfev), prices=fit2.x.tolist(),
                    residual=smooth(fit2.x).tolist(), message=str(fit2.message)))
                if np.max(abs(smooth(fit2.x))) <= 1e-6:
                    p = fit2.x
            gap = engine.transition_gap(p)
            if abs(gap) <= 1e-10:
                first = float(np.max(abs(kink_residual(engine, p, game['charge'])))) <= 1e-5
            else:
                first = float(np.max(abs(projected_gradient(p, engine.gradient(p, game['charge']),
                              engine.case.price_lower, engine.case.price_upper)))) <= 1e-6
            candidate = dict(prices=p.tolist(), first_order_ok=bool(first),
                transition_gap=float(gap), oriented_kink_residual=kr,
                work_price=kd['work_price'].tolist(), left=kd['left'].tolist(), right=kd['right'].tolist(),
                old_direction_wrong_platforms=[i+1 for i, d in enumerate(kd['work_price']) if d > 0],
                numerical_accepted=False)
            attempt['candidate'] = candidate
            state = snapshot(engine, p, engine.outcome(p))
            candidate['state'] = state
            # Both grids and bracketed roots remain diagnostics, never a proof.
            audit = audit_candidate(engine, p, game['charge'], deadline=deadline)
            candidate['finite_audit'] = audit
            candidate['numerical_accepted'] = bool(first and audit['epsilon_absolute'] <= 1e-6
                and max(c['native_profit_error'] for c in audit['coordinates']) <= 1e-6
                and audit['refinement_stable'])
            if candidate['numerical_accepted']:
                row['candidates'].append(candidate)
        except Exception as exc:
            attempt['error'] = repr(exc)
            row['errors'].append(dict(start=label, error=repr(exc)))
        attempt['evaluations'] = engine.budget.used
        attempt['choice_calls'] = engine.budget.choice_calls
        append(folder/'attempts.jsonl', dict(game=gid, event='finish', **attempt))
        dump(folder/('game_'+gid.replace(':', '_')+'.json'), row)
        if perf_counter() >= begin+plan['game_seconds']:
            break
    if row['candidates']:
        chosen = min(row['candidates'], key=lambda c:(float(np.linalg.norm(np.array(c['prices'])-anchor)),
                                                      tuple(c['prices'])))
        row.update(selected=chosen, numerical_accepted=True)
    row['elapsed_seconds'] = perf_counter()-begin
    dump(folder/('game_'+gid.replace(':', '_')+'.json'), row)
    return row


def interpret_saved(data, raw):
    """Pure arithmetic recovery of completed audits; preserve all raw failures."""
    rows = deepcopy(raw)
    for row in rows:
        _, anchor, _ = starts_for(data, row['game'])
        row['raw_numerical_accepted'] = row['numerical_accepted']
        row['raw_acceptance_error_preserved'] = True
        row['candidates'] = []
        for attempt in row['attempts']:
            c = attempt.get('candidate')
            if not c or 'finite_audit' not in c:
                continue
            audit = c['finite_audit']
            agreement = max(v['native_profit_error'] for v in audit['coordinates'])
            c['native_profit_error_max_from_coordinates'] = agreement
            c['numerical_accepted'] = bool(c['first_order_ok'] and audit['epsilon_absolute'] <= 1e-6
                and agreement <= 1e-6 and audit['refinement_stable'])
            if c['numerical_accepted']:
                row['candidates'].append(c)
        row['selected'] = min(row['candidates'], key=lambda c:(
            float(np.linalg.norm(np.array(c['prices'])-anchor)), tuple(c['prices']))) if row['candidates'] else None
        row['numerical_accepted'] = bool(row['selected'])
        row['interpretation'] = 'Saved coordinate-error maximum; no new optimizer, closure or BR audit'
    return rows


def corrected_games(data, new):
    optout = {g['game']:g for g in data['optout']}
    additions = {g['game']:g for g in new}
    rows = []
    for old in data['games']:
        states = []
        source = 'unresolved'
        excluded = []
        if old['game'] in additions and additions[old['game']]['numerical_accepted']:
            states = [c['state'] for c in additions[old['game']]['candidates']]
            selected = additions[old['game']]['selected']['state']
            source = 'new_oriented_kink'
        elif old['game'] in optout:
            replacement = optout[old['game']]
            if replacement['selected']:
                states = [replacement['state']]
                excluded = [r['platform']-1 for r in replacement['exclusion_proof']]
                source = 'saved_optout'
            selected = states[0] if states else None
        else:
            if old['selected']:
                states = [old['diagnostic']['state']]
                source = 'saved_fixed_positive_profit'
            selected = states[0] if states else None
        rows.append(dict(game=old['game'], menu=old['menu'], charge=old['charge'],
                         states=states, selected=selected, source=source, excluded=excluded))
    return rows


def menu_reports(data, new):
    games = corrected_games(data, new)
    reports = []
    for menu in data['spec']['menus']:
        resource = menu['case']['platform_resource']
        if max(resource)/min(resource) < 4.5-1e-12:
            continue
        items = sorted([g for g in games if g['menu'] == menu['id']], key=lambda g:g['charge'])
        engine = make_engine(menu)
        env = Envelope(engine)
        for kappa in (0., .5, 1.):
            points = []
            for g in items:
                values = [saved_surplus(s, engine, kappa) for s in g['states']]
                bounds = env.welfare_box(kappa)
                selected_value = saved_surplus(g['selected'], engine, kappa) if g['selected'] else None
                points.append(dict(game=g['game'], charge=g['charge'], known=bool(values), source=g['source'],
                    candidate_TS=[str(midpoint(v)) for v in values],
                    candidate_TS_intervals=[v.serialized() for v in values],
                    lower=str(min(v.lo for v in values)) if values else bounds['lower'],
                    upper=str(max(v.hi for v in values)) if values else bounds['upper'],
                    rigorous_unresolved_welfare=bounds if not values else None,
                    selected_TS=str(midpoint(selected_value)) if selected_value else None,
                    selected_TS_interval=selected_value.serialized() if selected_value else None))
            policy = policy_interval(points)
            selected_points = [p for p in points if p['known']]
            winner = max(selected_points, key=lambda p:(Decimal(p['selected_TS']), -p['charge']))
            reports.append(dict(menu=menu['id'], kappa=kappa, planned_charges=menu['charges'],
                points=points, **policy, selected_best_known_charge=winner['charge'],
                selected_optimal_charge=winner['charge'] if policy['complete'] else None,
                interpretation='Finite discovered candidates; unknown boxes conditional on Nash existence; not entire Nash set'))
    return reports


def audit_targets(data, new):
    games = corrected_games(data, new)
    targets = []
    strong = [m for m in data['spec']['menus'] if max(m['case']['platform_resource']) /
              min(m['case']['platform_resource']) >= 4.5-1e-12]
    for menu in strong:
        known = [g for g in games if g['menu'] == menu['id'] and g['selected']]
        winner = max(known, key=lambda g:(g['selected']['TS'], -g['charge']))
        nc = next((g for g in known if g['charge'] == 0), None)
        for role, g in [('optimal_known', winner), ('NC', nc)]:
            if g:
                targets.append(dict(menu=menu['id'], game=g['game'], role=role, charge=g['charge'],
                                    state=g['selected'], excluded=g['excluded']))
    for g in new:
        for c in g['candidates']:
            targets.append(dict(menu=g['menu'], game=g['game'], role='new_candidate',
                                charge=g['charge'], state=c['state'], excluded=[]))
    unique = {}
    for t in targets:
        key = (t['game'], tuple(t['state']['prices']), tuple(t['excluded']))
        if key in unique:
            unique[key]['roles'].append(t['role'])
        else:
            unique[key] = dict(t, roles=[t['role']])
    return list(unique.values())


def run(root, folder):
    root, folder = Path(root), Path(folder)
    verify_freeze(root, folder, include_main=False)
    if (folder/'execution_started.json').exists():
        raise FileExistsError('No reruns or extra starts in the frozen batch')
    data, plan = read(folder/'inputs.json'), read(folder/'design.json')
    dump(folder/'execution_started.json', dict(max_games=5, max_starts=3))
    games, wall = [], 0.
    for gid in data['missing_ids']:
        begin = perf_counter()
        command = [sys.executable, str(root/'scripts/run_review13_equilibrium.py'),
                   '--folder', str(folder), '--child', gid]
        error = None
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                    timeout=plan['hard_game_seconds'])
            if result.returncode:
                error = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as exc:
            error = dict(kind='hard_timeout', seconds=plan['hard_game_seconds'], detail=str(exc))
        elapsed = perf_counter()-begin
        wall += elapsed
        path = folder/('game_'+gid.replace(':', '_')+'.json')
        if path.exists():
            row = read(path)
        else:
            original = next(g for g in data['games'] if g['game'] == gid)
            row = dict(game=gid, menu=original['menu'], charge=original['charge'],
                       candidates=[], selected=None, numerical_accepted=False, attempts=[])
        row['child_process_seconds'] = elapsed
        row['child_error'] = error
        games.append(row)
        append(folder/'games.jsonl', row)
        print(json.dumps(dict(game=gid, accepted=row['numerical_accepted'], seconds=elapsed,
                              errors=row.get('errors', []), child_error=error)), flush=True)
    dump(folder/'solve_budget.json', dict(games=len(games), starts=sum(len(g['attempts']) for g in games),
         optimizer_calls=sum(len(a['fits']) for g in games for a in g['attempts']),
         child_wall_seconds=wall, max_seconds=450, within_budget=wall <= 450,
         failures=[dict(game=g['game'], start=a['start'], error=a.get('error'),
                         accepted=a.get('candidate', {}).get('numerical_accepted', False))
                   for g in games for a in g['attempts'] if not a.get('candidate', {}).get('numerical_accepted')]))
    targets = audit_targets(data, games)
    audit_begin = perf_counter()
    for t in targets:
        menu = next(m for m in data['spec']['menus'] if m['id'] == t['menu'])
        engine = OptOutEngine(MarketCase(**menu['case']), t['excluded']) if t['excluded'] else make_engine(menu)
        env = Envelope(engine)
        result = dict(game=t['game'], roles=t['roles'], excluded=t['excluded'],
                      prices=t['state']['prices'], charge=t['charge'], coordinates=[])
        for platform in env.active:
            seconds = plan['coordinate_interval_seconds'] if perf_counter()-audit_begin < plan['envelope_seconds'] else 0.
            try:
                result['coordinates'].append(env.unilateral(t['state']['prices'], t['charge'], platform,
                    max_leaves=plan['max_interval_leaves'], seconds=seconds))
            except Exception as exc:
                # Universal cap for positive profit covers every fare and root.
                margin = Interval(engine.case.price_upper)-env.c[platform]-Interval(t['charge'])*env.a[platform]
                maxdemand = min(env.N.hi, (env.K/env.a[platform]).hi)
                upper = (Interval(max(Decimal(0), margin.hi))*Interval(maxdemand)).hi
                result['coordinates'].append(dict(platform=platform+1, error=repr(exc),
                    fare_domain=[engine.case.price_lower, engine.case.price_upper], profit_upper=str(upper),
                    baseline_lower=None, gain_upper=None, coverage_complete=True,
                    epsilon_nash_closed=False, fallback='capacity/mass positive-margin universal bound'))
        result['epsilon_nash_closed'] = all(c['epsilon_nash_closed'] for c in result['coordinates'])
        append(folder/'profit_bounds.jsonl', result)
        print(json.dumps(dict(audited=t['game'], roles=t['roles'],
             gain_upper=[c['gain_upper'] for c in result['coordinates']], closed=result['epsilon_nash_closed'])), flush=True)
    dump(folder/'envelope_budget.json', dict(elapsed_seconds=perf_counter()-audit_begin,
         targets=len(targets), note='No Nash optimizers; coarse bounds retained after refinement deadline'))
    dump(folder/'counterexample.json', counterexample())
    dump(folder/'menus.json', menu_reports(data, games))
    return replay(folder)


def replay(folder):
    folder = Path(folder)
    if (folder/'summary.json').exists() and read(folder/'summary.json').get('schema') == 3:
        return validate_delivery_v3(folder)
    if (folder/'summary.json').exists() and read(folder/'summary.json').get('schema') == 2:
        return validate_delivery(folder)
    gamefile = folder/('interpreted_games.jsonl' if (folder/'interpreted_games.jsonl').exists() else 'games.jsonl')
    data, games = read(folder/'inputs.json'), jsonl(gamefile)
    boundfile = folder/('final_profit_bounds.jsonl' if (folder/'final_profit_bounds.jsonl').exists() else 'profit_bounds.jsonl')
    bounds = jsonl(boundfile) if boundfile.exists() else []
    menus = menu_reports(data, games)
    summary = dict(schema=1, planned_original_games=268, original_accepted=263,
        attempted_new_games=len(games), new_numerically_accepted=sum(g['numerical_accepted'] for g in games),
        new_starts=sum(len(g['attempts']) for g in games), solve_budget=read(folder/'solve_budget.json'),
        strong_menus=10, original_incomplete_menus=4, original_missing_NC=3,
        complete_menus_after=sum(m['complete'] for m in menus if m['kappa'] == 0),
        audited_candidates=len(bounds), entire_domain_small_epsilon=sum(b['epsilon_nash_closed'] for b in bounds),
        full_Nash_set_enumerated=False, nonexistence_claim=False,
        result_status='Numerical completion and rigorous conservative profit bounds are separate; small epsilon may remain open')
    dump(folder/'summary.json', summary)
    return summary


def validate_delivery(folder):
    """Read-only v2 replay: verify bytes, coverage and arithmetic interpretation."""
    folder = Path(folder)
    manifest = read(folder/'manifest.json')
    for name, expected in manifest['files'].items():
        if digest(folder/name) != expected:
            raise ValueError('Delivery hash mismatch: '+name)
    source = Path(manifest['v1_root'])
    if {str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()} != manifest['v1_sha256']:
        raise ValueError('v1 is not byte-identical to the preserved arithmetic-export input')
    data = read(folder/'inputs.json')
    expected = interpret_saved(data, jsonl(folder/'raw_v1/games.jsonl'))
    if expected != jsonl(folder/'interpreted_games.jsonl'):
        raise ValueError('Saved-state arithmetic interpretation differs')
    bounds = jsonl(folder/'final_profit_bounds.jsonl')
    for b in bounds:
        for coordinate in b['coordinates']:
            if not coordinate['coverage_complete'] or coordinate.get('gain_upper') is None:
                raise ValueError('Uncovered or silently skipped unilateral interval')
            if decimal(coordinate['gain_upper']) < 0:
                raise ValueError('Negative reported nonnegative gain bound')
    if len(bounds) != 31 or sum(len(b['coordinates']) for b in bounds) != 89:
        raise ValueError('Expected 31 target vectors and 89 active unilateral coordinates')
    return read(folder/'summary.json')


def reconcile(root, folder):
    """No Nash solving. Retain failed raw output; audit saved recovered candidates."""
    root, folder = Path(root), Path(folder)
    if (folder/'postprocess_started.json').exists():
        raise FileExistsError('Interpretation/audit is one-shot')
    hashes = read(folder/'freeze.json')
    for name, value in hashes['source_sha256'].items():
        if name not in OWNED and not name.endswith('/MAIN.tex') and digest(root/name) != value:
            raise ValueError('Historical input/source changed: '+name)
    data = read(folder/'inputs.json')
    raw = jsonl(folder/'games.jsonl')
    rows = interpret_saved(data, raw)
    rawhashes = {p.name:digest(p) for p in folder.glob('*.json*')}
    dump(folder/'postprocess_started.json', dict(no_new_solves=True, raw_sha256=rawhashes,
         errors_preserved=['14 KeyError(native_profit_error) after completed audits',
                           '1 genuine offgrid_4 fixed22 rejection',
                           'Counterexample np.bool JSON serialization failure after raw envelope run'],
         postprocess_source_sha256={p:digest(root/p) for p in OWNED}))
    for name in OWNED:
        destination = folder/'postprocess_source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, destination)
    for r in rows:
        append(folder/'interpreted_games.jsonl', r)
    dump(folder/'menus.json', menu_reports(data, rows))
    dump(folder/'counterexample.json', counterexample())
    old_bounds = jsonl(folder/'profit_bounds.jsonl')
    lookup = {(b['game'], tuple(b['prices']), tuple(b['excluded'])):b for b in old_bounds}
    targets = audit_targets(data, rows)
    # New selected low-charge points first, then other recovered starts. Old
    # complete-menu candidates reuse their already-computed sound enclosures.
    targets.sort(key=lambda t:(0 if t['role'] == 'NC' else 1 if 'new_candidate' in t['roles'] else 2,
                               t['game'], tuple(t['state']['prices'])))
    plan = read(folder/'design.json')
    spent = read(folder/'envelope_budget.json')['elapsed_seconds']
    begin = perf_counter()
    for t in targets:
        key = (t['game'], tuple(t['state']['prices']), tuple(t['excluded']))
        if key in lookup:
            result = dict(lookup[key], roles=t['roles'], reused_raw_bound=True)
        else:
            menu = next(m for m in data['spec']['menus'] if m['id'] == t['menu'])
            engine = make_engine(menu)
            env = Envelope(engine)
            result = dict(game=t['game'], roles=t['roles'], prices=t['state']['prices'],
                          charge=t['charge'], excluded=[], coordinates=[])
            for i in env.active:
                seconds = min(2., max(0., plan['envelope_seconds']-spent-(perf_counter()-begin)))
                result['coordinates'].append(env.unilateral(t['state']['prices'], t['charge'], i,
                        max_leaves=plan['max_interval_leaves'], seconds=seconds))
            result['epsilon_nash_closed'] = all(v['epsilon_nash_closed'] for v in result['coordinates'])
        append(folder/'final_profit_bounds.jsonl', result)
        print(json.dumps(dict(game=t['game'], roles=t['roles'], reused=result.get('reused_raw_bound', False),
              gain_upper=[v['gain_upper'] for v in result['coordinates']])), flush=True)
    dump(folder/'postprocess_budget.json', dict(new_Nash_games=0, new_starts=0,
         recovered_numerical_games=sum(r['numerical_accepted'] for r in rows),
         recovered_numerical_candidates=sum(len(r['candidates']) for r in rows),
         raw_envelope_seconds=spent, postprocess_seconds=perf_counter()-begin,
         refinement_seconds_limit=plan['envelope_seconds'],
         mandatory_coarse_bounds_retained_after_refinement_deadline=True,
         global_welfare_boxes=[dict(menu=m['id'], boxes=[Envelope(make_engine(m)).welfare_box(k)
            for k in (0., .5, 1.)]) for m in data['spec']['menus']
            if m['id'] in {r['menu'] for r in rows}]))
    summary = replay(folder)
    dump(folder/'final_manifest.json', dict(files={str(p.relative_to(folder)):digest(p)
         for p in folder.rglob('*') if p.is_file() and p.name != 'final_manifest.json'},
         historical_hashes_unchanged=True, raw_execution_sha256=rawhashes))
    return summary


def capacity_cap(engine, charge, raw):
    """No root or search: y_m <= min(N,K/a_m) for all feasible responses."""
    result = deepcopy(raw)
    i = raw['platform']-1
    mass = sum((Interval(x) for x in engine.demand_base), Interval(0))
    ymax = min(mass.hi, (Interval(engine.capacity)/Interval(engine.resource[i])).hi)
    margin = Interval(engine.case.price_upper)-Interval(engine.cost[i])-Interval(charge)*Interval(engine.resource[i])
    cap = (Interval(max(Decimal(0), margin.hi))*Interval(ymax)).hi
    upper = min(decimal(raw['profit_upper']), cap)
    result['original_envelope_profit_upper'] = raw['profit_upper']
    result['capacity_mass_profit_upper'] = str(cap)
    result['profit_upper'] = str(upper)
    if raw['baseline_lower'] is not None:
        gain = max(Decimal(0), UP.subtract(upper, decimal(raw['baseline_lower'])))
        result['gain_upper'] = str(gain)
        result['epsilon_nash_closed'] = gain <= Decimal('1e-6')
    result['bound_tightening'] = 'min(saved whole-domain upper, capacity/mass positive-margin upper); no new search'
    result['unresolved'] = not result['epsilon_nash_closed']
    return result


def failure_accounting(rows):
    technical, rejected, accepted = [], [], 0
    for row in rows:
        for attempt in row['attempts']:
            if attempt.get('error'):
                technical.append(dict(game=row['game'], start=attempt['start'], error=attempt['error']))
            candidate = attempt.get('candidate', {})
            if candidate.get('numerical_accepted'):
                accepted += 1
            elif 'finite_audit' in candidate:
                rejected.append(dict(game=row['game'], start=attempt['start'],
                    first_order_ok=candidate['first_order_ok'],
                    finite_gain=candidate['finite_audit']['epsilon_absolute'],
                    fit_terminations=[dict(kind=f['kind'], success=f['success'], status=f['status'],
                                           nfev=f['nfev']) for f in attempt['fits']]))
    return dict(technical_acceptance_exceptions=len(technical), technical_exceptions=technical,
                scientifically_rejected_starts=len(rejected), genuine_rejections=rejected,
                numerically_accepted_starts=accepted,
                field_error_is_not_nonconvergence=True, nonexistence_inference=False)


def export_v2(root, source, destination):
    """Arithmetic-only final delivery, leaving the entire v1 tree untouched."""
    root, source, destination = Path(root), Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('Final arithmetic export cannot overwrite an existing directory')
    start = perf_counter()
    v1_before = {str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
    data = read(source/'inputs.json')
    rows = interpret_saved(data, jsonl(source/'games.jsonl'))
    destination.mkdir(parents=True)
    for name in ('inputs.json', 'design.json', 'freeze.json', 'attempts.jsonl', 'games.jsonl',
                 'solve_budget.json', 'envelope_budget.json', 'profit_bounds.jsonl'):
        dst = destination/'raw_v1'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, dst)
    for name in OWNED:
        dst = destination/'execution_source'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, dst)
    dump(destination/'schema.json', dict(result_schema(),
        delivery='v2 arithmetic-only export; raw_v1 preserves all technical errors',
        additional_files=['interpreted_games.jsonl', 'failure_accounting.json',
                          'final_profit_bounds.jsonl', 'menus.json', 'global_welfare_boxes.json',
                          'missing_states.csv', 'menus.csv', 'solve_budget.json', 'manifest.json']))
    dump(destination/'inputs.json', data)
    for row in rows:
        append(destination/'interpreted_games.jsonl', row)
    failures = failure_accounting(rows)
    dump(destination/'failure_accounting.json', failures)
    finalbounds = []
    for saved in jsonl(source/'final_profit_bounds.jsonl'):
        menu = next(m for m in data['spec']['menus'] if saved['game'].startswith(m['id']+':'))
        engine = OptOutEngine(MarketCase(**menu['case']), saved['excluded']) if saved['excluded'] else make_engine(menu)
        result = dict(saved, coordinates=[capacity_cap(engine, saved['charge'], c) for c in saved['coordinates']])
        result['epsilon_nash_closed'] = all(c['epsilon_nash_closed'] for c in result['coordinates'])
        result['unresolved'] = not result['epsilon_nash_closed']
        finalbounds.append(result)
        append(destination/'final_profit_bounds.jsonl', result)
    menus = menu_reports(data, rows)
    globalboxes = []
    for menu in data['spec']['menus']:
        if max(menu['case']['platform_resource'])/min(menu['case']['platform_resource']) < 4.5-1e-12:
            continue
        env = Envelope(make_engine(menu))
        for kappa in (0., .5, 1.):
            box = env.welfare_box(kappa)
            globalboxes.append(dict(menu=menu['id'], **box,
                all_charges=menu['charges'], full_Nash_possible_optimal_charges=menu['charges'],
                note='Universal response enclosure; no Nash-specific price exclusion. Existence remains conditional.'))
            report = next(r for r in menus if r['menu'] == menu['id'] and r['kappa'] == kappa)
            report['global_response_welfare_box'] = box
            report['full_Nash_possible_optimal_charges'] = menu['charges']
            report['full_Nash_policy_closed'] = False
            report['complete_scope'] = 'All planned charges have numerical candidates, not exhaustive Nash sets'
    dump(destination/'menus.json', menus)
    dump(destination/'global_welfare_boxes.json', globalboxes)
    dump(destination/'counterexample.json', counterexample())
    rawbudget = read(source/'solve_budget.json')
    budget = dict(new_games=5, starts=15, optimizer_calls=rawbudget['optimizer_calls'],
        raw_child_wall_seconds=rawbudget['child_wall_seconds'], export_new_games=0,
        export_new_starts=0, export_new_optimizer_calls=0,
        cumulative_new_solve_seconds=rawbudget['child_wall_seconds'], limit_seconds=450,
        max_game_seconds=max(r['child_process_seconds'] for r in rows),
        technical_acceptance_exceptions=failures['technical_acceptance_exceptions'],
        scientifically_rejected_starts=failures['scientifically_rejected_starts'],
        within_budget=rawbudget['child_wall_seconds'] <= 450)
    dump(destination/'solve_budget.json', budget)
    missing = []
    for g in rows:
        s = g['selected']['state']
        missing.append(dict(game=g['game'], charge=g['charge'], candidates=len(g['candidates']),
             selected_TS=s['TS'], candidate_TS_min=min(c['state']['TS'] for c in g['candidates']),
             candidate_TS_max=max(c['state']['TS'] for c in g['candidates']),
             selected_prices=json.dumps(s['prices']), wait=s['wait_minutes'], rho=s['rho'],
             served=s['served'], max_finite_gain=max(c['finite_audit']['epsilon_absolute'] for c in g['candidates']),
             global_epsilon_closed=False))
    policyrows = [dict(menu=r['menu'], kappa=r['kappa'], complete=r['complete'],
        NC_min=r['NC_interval'][0], NC_max=r['NC_interval'][1],
        selected_charge=r['selected_optimal_charge'], possible_discovered_charges=json.dumps(r['possible_optimal_charges']),
        optimal_TS_min=r['optimum_welfare_interval'][0], optimal_TS_max=r['optimum_welfare_interval'][1],
        gain_min=r['gain_over_NC_interval'][0], gain_max=r['gain_over_NC_interval'][1],
        full_Nash_policy_closed=False) for r in menus]
    for name, values in [('missing_states.csv', missing), ('menus.csv', policyrows)]:
        with (destination/name).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = dict(schema=2, planned_original_games=268, original_accepted=263, new_numerically_accepted=5,
        numerically_accepted_starts=14, scientifically_rejected_starts=1, technical_exceptions=14,
        strong_menus=10, original_incomplete_menus=4, numerically_complete_menus=10,
        original_missing_NC=3, new_candidates=14, audited_candidates=len(finalbounds),
        audited_coordinates=sum(len(b['coordinates']) for b in finalbounds),
        entire_domain_small_epsilon=sum(b['epsilon_nash_closed'] for b in finalbounds),
        every_audit_has_full_fare_upper=all(c['coverage_complete'] for b in finalbounds for c in b['coordinates']),
        all_equilibria_enumerated=False, nonexistence_claim=False,
        conclusion='Five numerical gaps repaired via oriented kink sides. Whole-fare upper bounds are sound but too wide; global Nash/policy closure remains open.',
        solve_budget=budget, export_seconds=perf_counter()-start)
    dump(destination/'summary.json', summary)
    v1_after = {str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
    if v1_before != v1_after:
        raise ValueError('Original v1 changed during arithmetic export')
    dump(destination/'manifest.json', dict(v1_root=str(source), v1_sha256=v1_before,
        v1_untouched=True, no_new_fits=True,
        files={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()},
        historical_source_hashes=read(source/'freeze.json')['source_sha256']))
    return summary


def canonical_menus(data, rows):
    """Correct ground credit and outward arithmetic; no state error claim."""
    reports = menu_reports(data, rows)
    for report in reports:
        menu = next(m for m in data['spec']['menus'] if m['id'] == report['menu'])
        report.update(global_response_welfare_box=Envelope(make_engine(menu)).welfare_box(report['kappa']),
            full_Nash_possible_optimal_charges=menu['charges'], full_Nash_policy_closed=False,
            complete_scope='All charges have numerical candidates; not an exhaustive Nash correspondence',
            accounting='W^(TS,kappa) = TS + kappa*p0*ground',
            interval_scope='Outward arithmetic on saved numerical states; does not enclose their solver error')
    return reports


def export_v3(root, source, destination):
    """Correct P1/P2 by arithmetic only; keep every v1/v2 byte unchanged."""
    root, source, destination = Path(root), Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('Canonical v3 is immutable')
    before = {str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
    prior = read(source/'manifest.json')
    for name, expected in prior['files'].items():
        if digest(source/name) != expected:
            raise ValueError('v2 arithmetic source changed: '+name)
    data = read(source/'inputs.json')
    raw = jsonl(source/'raw_v1/games.jsonl')
    rows = interpret_saved(data, raw)
    menus = canonical_menus(data, rows)
    destination.mkdir(parents=True)
    for name, obj in [('inputs.json', data), ('schema.json', dict(result_schema(),
            schema_version=3, canonical=True, bound_endpoint_type='decimal string',
            candidate_TS='Decimal midpoint display only; candidate_TS_intervals governs comparisons',
            historical_error='v1/v2 kappa>0 credit sign and float-policy endpoints are invalid',
            accounting='TS + kappa*p0*ground', standalone_replay=True)),
            ('menus.json', menus), ('failure_accounting.json', failure_accounting(rows)),
            ('counterexample.json', counterexample())]:
        dump(destination/name, obj)
    for name in ('summary.json', 'menus.json', 'global_welfare_boxes.json'):
        dst = destination/'invalid_v2_arithmetic'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, dst)
    for name, values in [('raw_games.jsonl', raw), ('interpreted_games.jsonl', rows),
                         ('profit_bounds.jsonl', jsonl(source/'final_profit_bounds.jsonl'))]:
        for value in values:
            append(destination/name, value)
    globalboxes = [dict(menu=m['menu'], kappa=m['kappa'], **{
        k:v for k,v in m['global_response_welfare_box'].items() if k != 'kappa'},
        full_Nash_possible_optimal_charges=m['full_Nash_possible_optimal_charges']) for m in menus]
    dump(destination/'global_welfare_boxes.json', globalboxes)
    budget = dict(read(source/'solve_budget.json'), v3_new_games=0, v3_new_starts=0,
                  v3_new_optimizer_calls=0, v3_new_scalar_closure_solves=0, v3_new_BR_audits=0)
    dump(destination/'solve_budget.json', budget)
    # Include the imported dependency closure, not just the task-A source. The
    # replay runner can execute here without referring to external result trees.
    names = list(OWNED)
    names += [name for name in prior['historical_source_hashes']
              if name.startswith('experiments/') and name.endswith('.py')]
    names = sorted(set(names))
    source_hashes = {}
    for name in names:
        source_hashes[name] = digest(root/name)
        dst = destination/'execution_source'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, dst)
    summary = dict(schema=3, canonical=True, replaces_invalid_arithmetic='v1/v2 kappa>0 menus and welfare bounds',
        ground_credit_sign='+', policy_endpoint_type='outward Decimal strings',
        new_games=5, starts=15, numerically_accepted_games=5, numerically_accepted_starts=14,
        scientifically_rejected_starts=1, technical_acceptance_exceptions=14,
        original_strong_menus_complete=6, numerically_complete_strong_menus=10,
        repaired_NC_states=3, saved_new_candidates=14,
        audited_candidates=31, audited_coordinates=89, small_global_epsilon_closed=0,
        nonexistence_claim=False, all_equilibria_enumerated=False, full_Nash_policy_closed=False,
        solve_budget=budget,
        statement='Five numerical gaps repaired. Whole-fare bounds remain too wide for small global epsilon. Corrected positive ground credit; outward policy arithmetic only, not saved-state error inclusion.')
    dump(destination/'summary.json', summary)
    policyrows = [dict(menu=m['menu'], kappa=m['kappa'], complete=m['complete'],
        NC_min=m['NC_interval'][0], NC_max=m['NC_interval'][1],
        selected_charge=m['selected_optimal_charge'],
        possible_discovered_charges=json.dumps(m['possible_optimal_charges']),
        optimal_TS_min=m['optimum_welfare_interval'][0], optimal_TS_max=m['optimum_welfare_interval'][1],
        gain_min=m['gain_over_NC_interval'][0], gain_max=m['gain_over_NC_interval'][1],
        full_Nash_policy_closed=False) for m in menus]
    missingrows = []
    for game in rows:
        for kappa in (0., .5, 1.):
            report = next(m for m in menus if m['menu'] == game['menu'] and m['kappa'] == kappa)
            point = next(p for p in report['points'] if p['charge'] == game['charge'])
            missingrows.append(dict(game=game['game'], kappa=kappa,
                candidates=len(game['candidates']), selected_TS=point['selected_TS'],
                candidate_TS_min=point['lower'], candidate_TS_max=point['upper'],
                global_epsilon_closed=False))
    for name, values in [('menus.csv', policyrows), ('missing_states.csv', missingrows)]:
        with (destination/name).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    after = {str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
    if before != after:
        raise ValueError('v2 changed during v3 arithmetic export')
    dump(destination/'manifest.json', dict(schema=3, source_sha256=source_hashes,
        v2_preserved=True, v2_source_sha256=before, v1_sha256=prior['v1_sha256'],
        archive_source_directory=str(source), no_external_result_dependency_for_replay=True,
        files={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()}))
    return summary


def validate_delivery_v3(folder):
    """Standalone and read-only: hashes, arithmetic and frozen code identity."""
    folder = Path(folder)
    manifest = read(folder/'manifest.json')
    for name, expected in manifest['files'].items():
        if digest(folder/name) != expected:
            raise ValueError('Canonical delivery hash differs: '+name)
    current_root = Path(__file__).resolve().parents[2]
    for name, expected in manifest['source_sha256'].items():
        if digest(folder/'execution_source'/name) != expected:
            raise ValueError('Saved execution source changed: '+name)
        if digest(current_root/name) != expected:
            raise ValueError('Running code differs from frozen execution source: '+name)
    data = read(folder/'inputs.json')
    rows = interpret_saved(data, jsonl(folder/'raw_games.jsonl'))
    if rows != jsonl(folder/'interpreted_games.jsonl'):
        raise ValueError('Saved-state interpretation differs')
    if canonical_menus(data, rows) != read(folder/'menus.json'):
        raise ValueError('Ground-credit/Decimal policy replay differs')
    bounds = jsonl(folder/'profit_bounds.jsonl')
    if len(bounds) != 31 or sum(len(b['coordinates']) for b in bounds) != 89:
        raise ValueError('Whole-domain bound target coverage differs')
    for target in bounds:
        for c in target['coordinates']:
            if not c['coverage_complete'] or c['gain_upper'] is None:
                raise ValueError('Uncovered fare region or unknown gain bound')
    return read(folder/'summary.json')
