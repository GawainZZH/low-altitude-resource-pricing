"""Frozen-menu, bounded full-interval unilateral response evidence.

Derivative brackets and explicit regime events are numerical searches, not
mathematical interval bounds. No concavity, dual bound, or pruning is assumed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from .config import MarketCase as CaseConfig
from .review_equilibrium import ReviewEngine
from .review_hard_capacity_repair import RepairEngine
from .review90_waiting import WaitingEngine


PARAMETERS = 'results/trb_review_parameters_v1/full_parameters.json'
SOFT = 'results/comprehensive_state_grid_v1_full/comprehensive_state_grid_charge_search_records.csv'
HARD = 'results/trb_review_hard_capacity_repair_v1/corrected_hard_charge_records.csv'
WAITING = 'results/trb_review90_waiting_v1/waiting_games.jsonl'
ANCHOR = 'results/calibrated_anchor_case_v1/anchor_charge_search_records.csv'
HARD_ATTEMPTS = 'results/trb_review_hard_capacity_repair_v1/full_candidates.jsonl'
MANUSCRIPT = 'submission_packages/TRA_V290_LaTeX_Project/MAIN.tex'
GAIN_TOL = 1e-5


class AuditBudgetExceeded(RuntimeError):
    pass


@dataclass
class FareBudget:
    limit: int = 5000
    deadline: float = float('inf')
    used: int = 0

    def spend(self, n):
        if self.used+n > self.limit or perf_counter() >= self.deadline:
            raise AuditBudgetExceeded('fare-evaluation or wall deadline exhausted')
        self.used += n


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_fares(values, tolerance=1e-7):
    result = []
    for value in sorted(float(x) for x in values):
        if not result or value-result[-1] > tolerance:
            result.append(value)
    return result


def segmented_search(evaluate, lower, upper, baseline, points=129, kinks=(),
                     max_evaluations=5000, deadline=float('inf')):
    """Find every sampled derivative sign bracket on each smooth segment.

    All extrema signs are retained, including minima. Even-multiplicity roots
    or unresolved pairs inside one grid cell can escape this finite search.
    """
    if points < 3 or not lower <= baseline <= upper or lower >= upper:
        raise ValueError('invalid fare interval, baseline, or grid')
    budget = FareBudget(max_evaluations, deadline)
    cache, trace = {}, []

    def batch(xs, purpose):
        xs = [float(x) for x in xs]
        new = list(dict.fromkeys(x for x in xs if x not in cache))
        if new:
            budget.spend(len(new))
            profits, derivatives = evaluate(np.asarray(new))
            if not np.all(np.isfinite([profits, derivatives])):
                raise ValueError('nonfinite profit or derivative')
            for x, v, d in zip(new, profits, derivatives):
                cache[x] = (float(v), float(d))
                trace.append(dict(fare=x, profit=float(v), derivative=float(d), purpose=purpose))
        return np.array([cache[x] for x in xs])

    breaks = [lower]+unique_fares(x for x in kinks if lower < x < upper)+[upper]
    grid = np.linspace(lower, upper, points)
    batch([baseline, *grid, *breaks], 'grid_endpoint_or_event')
    roots, brackets = [], []
    for segment, (a, b) in enumerate(zip(breaks[:-1], breaks[1:])):
        # A small one-sided offset prevents a derivative jump being labelled
        # as a smooth root. Event profit is always evaluated at the event.
        offset = min(1e-7, (b-a)*1e-4)
        left = a+offset if segment else a
        right = b-offset if segment < len(breaks)-2 else b
        xs = np.unique(np.r_[left, grid[(grid > left)&(grid < right)], right])
        ds = batch(xs, 'segment_probe')[:, 1]
        for i, d in enumerate(ds):
            if abs(d) < 1e-10:
                roots.append(float(xs[i]))
        for x, y, dx, dy in zip(xs[:-1], xs[1:], ds[:-1], ds[1:]):
            if dx*dy < 0:
                brackets.append([float(x), float(y)])
                root = brentq(lambda z: batch([z], 'stationary_root_probe')[0, 1],
                              x, y, xtol=2e-10, rtol=1e-13, maxiter=60)
                roots.append(float(root))
                batch([root], 'stationary_candidate')
    best = max(cache, key=lambda x: (cache[x][0], -abs(x-baseline), -x))
    base = cache[float(baseline)][0]
    return dict(points=points, interval=[lower, upper], baseline=base,
                best_fare=best, best_profit=cache[best][0], gain=max(0., cache[best][0]-base),
                stationary_roots=unique_fares(roots), stationary_brackets=brackets,
                kinks=breaks[1:-1], evaluations=budget.used, trace=trace,
                pruned_intervals=0, formal_interval_bound=False)


class VectorResponse:
    """NumPy own-fare closure with the original engine's effective parameters."""
    def __init__(self, engine, prices, charge, platform, budget=None):
        self.e, self.p = engine, np.asarray(prices, dtype=float)
        self.charge, self.m = float(charge), int(platform)
        self.budget = budget or FareBudget()
        self.waiting = isinstance(engine, WaitingEngine)
        self.hard = engine.kind == 'hard'
        self.cache = {}
        e = engine
        self.a = e.price_sensitivity/e.logit_scale
        self.b = e.reliability_sensitivity/e.logit_scale
        outside = (e.outside_affinity-e.price_sensitivity*e.case.outside_price
                   -e.time_sensitivity*e.case.outside_time)/e.logit_scale
        self.base = (e.platform_affinity-e.price_sensitivity[:, None]*self.p
                     -e.time_sensitivity[:, None]*e.time
                     -e.reliability_sensitivity[:, None]*e.base_loss)/e.logit_scale-outside[:, None]
        self.dual_beta = e.beta/e.logit_scale if self.waiting else np.ones_like(self.a)

    def choice(self, fares, rho, dual=0.):
        fares = np.atleast_1d(fares).astype(float)
        r = np.clip(np.broadcast_to(rho, fares.shape), 0., 1.45)
        h = .55*r**self.e.case.reliability_power+1.10*np.maximum(r-.88, 0.)**2
        u = np.broadcast_to(self.base, (len(fares), *self.base.shape)).copy()
        u[:, :, self.m] -= (fares-self.p[self.m])[:, None]*self.a
        u -= h[:, None, None]*self.b[None, :, None]*self.e.exposure
        u -= np.broadcast_to(dual, fares.shape)[:, None, None]*self.dual_beta[None, :, None]
        peak = np.maximum(0., u.max(axis=2))
        expu = np.exp(u-peak[:, :, None])
        prob = expu/(np.exp(-peak)+expu.sum(axis=2))[:, :, None]
        return np.einsum('g,ngj->nj', self.e.demand_base, prob), prob

    def fixed_rho_gap(self, fares, rho):
        self.budget.spend(len(np.atleast_1d(fares)))
        y, _ = self.choice(fares, rho)
        return y @ self.e.resource/self.e.capacity-rho

    def outcomes(self, fares):
        fares = np.atleast_1d(fares).astype(float)
        n, e = len(fares), self.e
        rho, dual = np.zeros(n), np.zeros(n)
        binding = np.zeros(n, dtype=bool)
        if self.hard:
            y, _ = self.choice(fares, 1.)
            binding = y @ e.resource > e.capacity
        free = ~binding
        if free.any():
            fs = fares[free]
            lo, hi = np.zeros(len(fs)), np.full(len(fs), 1. if self.hard else e.rmax)
            for _ in range(47):
                mid = (lo+hi)/2
                y, _ = self.choice(fs, mid)
                positive = y @ e.resource/e.capacity > mid
                lo, hi = np.where(positive, mid, lo), np.where(positive, hi, mid)
            rho[free] = (lo+hi)/2
        if binding.any():
            fs = fares[binding]
            lo, hi = np.zeros(len(fs)), np.ones(len(fs))
            for _ in range(20):
                y, _ = self.choice(fs, 1., hi)
                high = y @ e.resource > e.capacity
                if not high.any():
                    break
                hi = np.where(high, hi*2, hi)
            else:
                raise ValueError('unbracketed nonnegative capacity dual')
            for _ in range(47):
                mid = (lo+hi)/2
                y, _ = self.choice(fs, 1., mid)
                positive = y @ e.resource > e.capacity
                lo, hi = np.where(positive, mid, lo), np.where(positive, hi, mid)
            dual[binding], rho[binding] = (lo+hi)/2, 1.
        y, prob = self.choice(fares, rho, dual)
        return dict(demand=y, prob=prob, rho=rho, dual=dual, binding=binding,
                    residual=abs(y @ e.resource/e.capacity-rho))

    def evaluate(self, fares):
        fares = np.atleast_1d(fares).astype(float)
        fresh = list(dict.fromkeys(float(x) for x in fares if float(x) not in self.cache))
        if fresh:
            self.budget.spend(len(fresh))
            x, e, m = np.array(fresh), self.e, self.m
            o = self.outcomes(x)
            p, rho = o['prob'], o['rho']
            weighted = e.demand_base*self.a
            direct = np.einsum('g,ng,ngj->nj', weighted, p[:, :, m], p)
            direct[:, m] -= np.einsum('g,ng->n', weighted, p[:, :, m])
            centered = e.exposure-np.einsum('ngj,j->ng', p, e.exposure)[:, :, None]
            hp = np.where(rho < 1.45, .55*e.case.reliability_power*rho**(e.case.reliability_power-1)
                          +2.20*np.maximum(rho-.88, 0.), 0.)
            v = np.einsum('g,ngj,ngj->nj', -e.demand_base*self.b, p, centered)*hp[:, None]
            own = direct[:, m]+v[:, m]*(direct @ e.resource)/(e.capacity-v @ e.resource)
            mask = o['binding']
            if mask.any():
                z = np.einsum('g,ng,ngj->nj', e.demand_base*self.dual_beta,
                              1-p.sum(axis=2), p)
                own[mask] = (direct[:, m]-z[:, m]*(direct @ e.resource)/(z @ e.resource))[mask]
            # cost is already the effective per-passenger engine cost c_m.
            margin = x-e.cost[m]-self.charge*e.resource[m]
            profit = margin*o['demand'][:, m]
            gradient = o['demand'][:, m]+margin*own
            for i, value in enumerate(fresh):
                self.cache[value] = (float(profit[i]), float(gradient[i]))
        values = np.array([self.cache[float(x)] for x in fares])
        return values[:, 0], values[:, 1]

    def events(self, lower, upper, points):
        events = []
        grid = np.linspace(lower, upper, points)
        for rho in ([.88, 1.] if self.hard else [.88, 1.45]):
            gap = self.fixed_rho_gap(grid, rho)
            roots = [float(x) for x, g in zip(grid, gap) if abs(g) < 1e-12]
            for a, b, ga, gb in zip(grid[:-1], grid[1:], gap[:-1], gap[1:]):
                if ga*gb < 0:
                    roots.append(brentq(lambda x: self.fixed_rho_gap([x], rho)[0],
                                        a, b, xtol=2e-11, rtol=1e-13))
            events.extend(dict(fare=x, rho=rho,
                               kind='capacity_release' if rho == 1. else 'reliability_piece')
                          for x in unique_fares(roots))
        return events


def root_distance(a, b):
    if not a and not b:
        return 0.
    if not a or not b:
        return None
    return float(max(max(min(abs(x-y) for y in b) for x in a),
                     max(min(abs(x-y) for y in a) for x in b)))


def audit_candidate(engine, prices, charge, max_evaluations=5000, deadline=float('inf')):
    started = perf_counter()
    budget = FareBudget(max_evaluations, deadline)
    coordinates = []
    prices = np.asarray(prices, dtype=float)
    base = engine.outcome(prices)
    gap = float(engine.resource @ engine.choice(prices, 1.)[0]/engine.capacity-1)
    budget.spend(2)
    for m in range(len(prices)):
        response = VectorResponse(engine, prices, charge, m, budget)
        passes, event_passes = [], []
        for points in (129, 257):
            events = response.events(engine.case.price_lower, engine.case.price_upper, points)
            event_passes.append(events)
            passes.append(segmented_search(response.evaluate, engine.case.price_lower,
                engine.case.price_upper, prices[m], points, [e['fare'] for e in events],
                max_evaluations=max_evaluations, deadline=deadline))
        a, b = passes
        best = max(passes, key=lambda r: r['best_profit'])
        q = prices.copy()
        q[m] = best['best_fare']
        budget.spend(2)
        native_best = engine.profit(q, m, charge)
        native_base = engine.profit(prices, m, charge)
        native_gain = max(0., native_best-native_base)
        err = max(abs(native_best-best['best_profit']), abs(native_base-best['baseline']))
        normalized = native_gain/max(1., abs(native_base))
        coordinates.append(dict(platform=m+1, passes=passes, events=event_passes,
            gain_absolute=native_gain, gain_normalized=normalized,
            native_profit_error=err, native_baseline=native_base, native_best=native_best,
            gain_difference=abs(a['gain']-b['gain']),
            best_fare_difference=abs(a['best_fare']-b['best_fare']),
            root_count_difference=len(b['stationary_roots'])-len(a['stationary_roots']),
            root_hausdorff_difference=root_distance(a['stationary_roots'], b['stationary_roots']),
            event_count_difference=len(event_passes[1])-len(event_passes[0])))
    epsilon = max(c['gain_absolute'] for c in coordinates)
    agreement = max(c['native_profit_error'] for c in coordinates)
    stable = all(c['gain_difference'] <= 1e-7 and c['root_count_difference'] == 0
                 and c['event_count_difference'] == 0
                 and (c['root_hausdorff_difference'] or 0.) <= 1e-6 for c in coordinates)
    status = ('engine_disagreement' if agreement > 1e-6 else
              'positive_gain' if epsilon > GAIN_TOL else
              'refinement_unresolved' if not stable else 'epsilon_search_pass')
    return dict(prices=prices.tolist(), charge=float(charge), status=status,
                epsilon_absolute=epsilon,
                epsilon_normalized=max(c['gain_normalized'] for c in coordinates),
                rho=float(base.rho), binding=bool(base.binding), transition_gap=gap,
                near_capacity=bool(abs(gap) <= .002), coordinates=coordinates,
                evaluations=budget.used, elapsed_seconds=perf_counter()-started,
                refinement_stable=stable, formal_interval_bound=False)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_games(root):
    root = Path(root)
    raw = json.loads((root/PARAMETERS).read_text(encoding='utf-8'))
    cases = {v['dataclass']['name']: CaseConfig(**v['dataclass']) for v in raw['full_state_grid']}
    anchor = CaseConfig(**raw['anchor']['dataclass'])
    games = []
    for kind, source in [('soft', SOFT), ('hard', HARD)]:
        grouped = {}
        rows = read_csv(root/source)
        if len(rows) != 4950:
            raise ValueError(f'{kind}: expected 4950 frozen rows')
        for row in rows:
            case = cases[row['case']]
            signature = asdict(case)
            signature.pop('name')
            signature.pop('congestion_cost')
            key = (json.dumps(signature, sort_keys=True), float(row['charge']))
            p = [float(row[f'price_{j}']) for j in (1, 2, 3)]
            if key not in grouped:
                grouped[key] = dict(id=f'{kind}:{case.name.rsplit("_e", 1)[0]}:q{key[1]:g}',
                    case=case, kind=kind, charge=key[1], prices=p, replicas=[], branches=[])
            game = grouped[key]
            if max(abs(np.array(game['prices'])-p)) > 1e-8:
                raise ValueError('externality-replica price discrepancy')
            game['replicas'].append(row['case'])
        if len(grouped) != 990 or any(len(g['replicas']) != 5 for g in grouped.values()):
            raise ValueError('behavioral game coverage mismatch')
        games.extend(grouped.values())
    representatives = {(c.demand_scale, c.capacity_scale, c.reliability_power): c
                       for c in cases.values() if c.congestion_cost == 90}
    for row in read_jsonl(root/WAITING):
        case = representatives[tuple(row['profile'])]
        if not row.get('selected'):
            raise ValueError('frozen waiting menu contains an unselected game')
        games.append(dict(id=row['game'], case=case, kind=row['mode'],
                          charge=float(row['charge']), prices=row['selected']['prices'],
                          replicas=[case.name], branches=row.get('attempts', [])))
    for row in read_csv(root/ANCHOR):
        games.append(dict(id=f'anchor:q{float(row["charge"]):g}', kind='anchor', case=anchor,
                          charge=float(row['charge']), replicas=[anchor.name], branches=[],
                          prices=[float(row[f'price_{j}']) for j in (1, 2, 3)]))
    hard = {(g['case'].demand_scale, g['case'].capacity_scale,
             g['case'].reliability_power, g['charge']): g for g in games if g['kind'] == 'hard'}
    for row in read_jsonl(root/HARD_ATTEMPTS):
        if row['game'].startswith('coarse:'):
            profile, charge = row['game'].split(':')[1:]
            d, k, p = (float(x[1:]) for x in profile.split('_'))
            hard[d, k, p, float(charge[1:])]['branches'] = row.get('attempts', [])
    counts = Counter(g['kind'] for g in games)
    if counts != dict(soft=990, hard=990, native=198, equal_vot=33, equal_beta=33, anchor=11):
        raise ValueError(f'unexpected game coverage: {counts}')
    sources = [PARAMETERS, SOFT, HARD, WAITING, ANCHOR, HARD_ATTEMPTS]
    return games, sources


def make_engine(game):
    case, kind = game['case'], game['kind']
    if kind in ('soft', 'anchor'):
        return ReviewEngine(case)
    if kind == 'hard':
        return RepairEngine(case)
    beta = np.array(case.time_sensitivity)
    if kind == 'equal_beta':
        beta[:] = .038
    elif kind == 'equal_vot':
        beta = np.array(case.price_sensitivity)*(np.dot(case.base_demand, beta)/
                                                np.dot(case.base_demand, case.price_sensitivity))
    return WaitingEngine(case, beta=beta)


def branch_candidates(game):
    seen, candidates = [np.asarray(game['prices'])], []
    for index, attempt in enumerate(game['branches']):
        prices = attempt.get('prices')
        if prices is None or not np.all(np.isfinite(prices)):
            continue
        if any(np.max(abs(np.asarray(prices)-p)) <= 1e-6 for p in seen):
            continue
        seen.append(np.asarray(prices))
        candidates.append(dict(index=index, prices=prices,
            old_verified=bool(attempt.get('verified', attempt.get('accepted', False))),
            candidate_source=attempt.get('candidate_source', 'saved_attempt')))
    return candidates


def summarize(rows):
    result = dict(attempted=len(rows), formal_interval_bound=False, groups=[])
    for kind in sorted({r['kind'] for r in rows}):
        for role in ('menu', 'branch'):
            selected = [r for r in rows if r['kind'] == kind and r['role'] == role]
            if not selected:
                continue
            complete = [r for r in selected if 'epsilon_absolute' in r]
            result['groups'].append(dict(kind=kind, role=role, attempted=len(selected),
                completed=len(complete), statuses=dict(Counter(r['status'] for r in selected)),
                epsilon_absolute=max((r['epsilon_absolute'] for r in complete), default=None),
                epsilon_normalized=max((r['epsilon_normalized'] for r in complete), default=None),
                near_capacity_games=sum(r['near_capacity'] for r in complete),
                near_capacity_epsilon=max((r['epsilon_absolute'] for r in complete if r['near_capacity']), default=None),
                max_gain_refinement_difference=max((c['gain_difference'] for r in complete for c in r['coordinates']), default=None),
                max_root_hausdorff_difference=max((c['root_hausdorff_difference'] or 0. for r in complete for c in r['coordinates']), default=None),
                root_count_changed=sum(c['root_count_difference'] != 0 for r in complete for c in r['coordinates']),
                max_native_profit_error=max((c['native_profit_error'] for r in complete for c in r['coordinates']), default=None)))
    result['positive_gains'] = [dict(id=r['id'], role=r['role'], epsilon=r['epsilon_absolute'])
                               for r in rows if r.get('epsilon_absolute', 0.) > GAIN_TOL]
    return result
