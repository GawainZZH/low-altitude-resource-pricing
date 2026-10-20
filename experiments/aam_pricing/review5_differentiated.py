"""Strict-capacity welfare dominance on differentiated local Nash responses."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .review3_ground import load_parameters
from .review_equilibrium import ReviewEngine
from .review90_waiting import WaitingEngine


def welfare_balance(prices, costs, resource, demand_derivative, friction_derivative,
                    friction_value, kappa=0., ground_price=0.):
    p, c, a, dy = [np.asarray(v, dtype=float) for v in
                    (prices, costs, resource, demand_derivative)]
    if (p.ndim != 1 or not p.size or any(v.shape != p.shape for v in (c, a, dy))
            or any(not np.all(np.isfinite(v)) for v in (p, c, a, dy))
            or np.any(a <= 0) or not 0 <= kappa <= 1
            or not np.isfinite(ground_price) or ground_price < 0
            or not np.isfinite(friction_derivative) or not np.isfinite(friction_value)
            or friction_value < 0):
        raise ValueError('Finite conformable vectors and positive resources required')
    error = float(a @ dy)
    if abs(error) > 1.e-8*(1+np.sum(np.abs(a*dy))):
        raise ValueError('Strict resource conservation required')
    net = p-c-kappa*ground_price
    margins = net/a
    spread = float(np.ptp(margins))
    turnover = float(.5*np.sum(np.abs(a*dy)))
    relief = float(-friction_value*friction_derivative)
    allocation = float(net @ dy)
    correction = float(.5*(np.max(margins)+np.min(margins))*error)
    lower = relief-spread*turnover+correction
    return dict(relief=relief, allocation=allocation, spread=spread,
                turnover=turnover, lower_bound=lower,
                TS_derivative=relief+allocation, sufficient=lower > 0,
                resource_derivative_error=abs(error), conservation_correction=correction)


def local_response(engine, prices, charge, kappa=0., jacobian_step=2.e-4):
    p = np.array(prices, dtype=float)
    lo, hi = engine.case.price_lower, engine.case.price_upper
    if np.any(p-jacobian_step <= lo) or np.any(p+jacobian_step >= hi):
        raise ValueError('Local IFT audit requires strictly interior fares')
    o = engine.outcome(p)
    is_waiting = isinstance(engine, WaitingEngine)
    loss = o.wait_minutes if is_waiting else -o.logphi
    if not o.binding or loss <= 1.e-6:
        raise ValueError('A strictly positive access friction is required')
    g = engine.gradient(p, charge, o)
    if np.max(np.abs(g)) > 1.e-4:
        raise ValueError('Candidate stationarity residual exceeds 1e-4')
    active = ((p-lo < 1.e-6) & (g < -1.e-5)) | ((hi-p < 1.e-6) & (g > 1.e-5))
    free = ~active
    jac = np.empty((len(p), len(p)))
    for m in range(len(p)):
        step = np.zeros(len(p)); step[m] = jacobian_step
        probes = [engine.outcome(p+direction*step) for direction in (1, -1)]
        if any(not probe.binding or (probe.wait_minutes if is_waiting else -probe.logphi) <= 1.e-6
               for probe in probes):
            raise ValueError('Coordinate derivative crosses a capacity transition')
        jac[:, m] = (engine.gradient(p+step, charge, probes[0])-
                     engine.gradient(p-step, charge, probes[1]))/(2*jacobian_step)
    demand_jac = engine.jacobian(p, o)
    gcharge = -engine.resource*np.diag(demand_jac)
    pdot = np.zeros(len(p))
    if free.any():
        subsystem = jac[np.ix_(free, free)]
        condition = float(np.linalg.cond(subsystem))
        if not np.isfinite(condition) or condition > 1.e10:
            raise ValueError('Singular local free-fare Jacobian')
        pdot[free] = np.linalg.solve(subsystem, -gcharge[free])
    else:
        condition = 1.
    prob, mass, lam = o.prob, engine.demand_base, engine.price_sensitivity
    mu = engine.logit_scale
    b = engine.beta if is_waiting else np.full(len(mass), mu)
    theta = prob.sum(axis=1)
    weight = mass*lam/mu
    direct = np.einsum('g,gm,gj->mj', weight, prob, prob)-np.diag(weight@prob)
    friction_demand = -((mass*b/mu*(1-theta))[:, None]*prob).sum(axis=0)
    friction_price = -(engine.resource@direct)/float(engine.resource@friction_demand)
    fdot = float(friction_price@pdot)
    dy = demand_jac@pdot
    value = float(np.sum(mass*theta*b/lam))
    result = welfare_balance(p, engine.cost, engine.resource, dy, fdot, value,
                             kappa, engine.case.outside_price)
    h = 1.e-5
    values = []
    for direction in (-1, 1):
        pp = p+direction*h*pdot
        if np.any(pp <= lo) or np.any(pp >= hi):
            raise ValueError('Directional comparison crosses a fare boundary')
        oo = engine.outcome(pp)
        if not oo.binding or (oo.wait_minutes if is_waiting else -oo.logphi) <= 1.e-6:
            raise ValueError('Derivative comparison crosses a capacity transition')
        _, ts = engine.welfare(pp, outcome=oo)
        ground = float(mass.sum()-oo.demand.sum())
        values.append(ts+kappa*engine.case.outside_price*ground)
    finite = (values[1]-values[0])/(2*h)
    return dict(price_derivative=pdot.tolist(), friction_derivative=fdot,
                demand_derivative=dy.tolist(), balance=result, finite_derivative=finite,
                derivative_error=abs(finite-result['TS_derivative']),
                resource_derivative_error=abs(float(engine.resource@dy)),
                gradient_residual=float(np.max(np.abs(g[free]))) if free.any() else 0.,
                active_fares=int(active.sum()), jacobian_condition=condition,
                friction=loss)


def write_csv(path, rows):
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, names)
        writer.writeheader(); writer.writerows(rows)


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = ['results/trb_review_hard_capacity_repair_v1/corrected_hard_charge_records.csv',
              'results/trb_review90_waiting_v1/waiting_games.jsonl',
              'results/trb_review_parameters_v1/full_parameters.json']
    parameters = load_parameters(root)
    cases = parameters['cases']
    candidates = []
    with (root/source[0]).open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if cases[r['case']].congestion_cost != 90:
                continue
            if -np.log(float(r['hard_availability'])) <= 1.e-6:
                continue
            candidates.append(('availability', r['case'], float(r['charge']),
                               [float(r[f'price_{m}']) for m in (1, 2, 3)]))
    for line in (root/source[1]).read_text(encoding='utf-8').splitlines():
        g = json.loads(line)
        if g['mode'] != 'native' or g['waiting_outcome']['wait_minutes'] <= 1.e-6:
            continue
        case = next(c for c in cases.values() if c.congestion_cost == 90
                    and [c.demand_scale, c.capacity_scale, c.reliability_power] == g['profile'])
        candidates.append(('waiting', case.name, float(g['charge']), g['selected']['prices']))
    rows, failures = [], []
    for model, name, q, p in candidates:
        engine = WaitingEngine(cases[name]) if model == 'waiting' else ReviewEngine(cases[name], 'hard')
        for kappa in (0., .5, 1.):
            try:
                result = local_response(engine, p, q, kappa)
                repeat = local_response(engine, p, q, kappa, jacobian_step=1.e-4)
                row = dict(model=model, case=name, charge=q, kappa=kappa,
                           **result['balance'],
                           friction_derivative=result['friction_derivative'],
                           derivative_error=result['derivative_error'],
                           jacobian_step_error=max(abs(np.array(result['price_derivative'])-
                                                       np.array(repeat['price_derivative']))),
                           gradient_residual=result['gradient_residual'],
                           active_fares=result['active_fares'],
                           jacobian_condition=result['jacobian_condition'])
                for m, value in enumerate(result['price_derivative'], 1):
                    row[f'price_{m}_derivative'] = value
                rows.append(row)
            except (ValueError, np.linalg.LinAlgError) as exc:
                failures.append(dict(model=model, case=name, charge=q, kappa=kappa, error=str(exc)))
    summary = []
    for model in ('availability', 'waiting'):
        for kappa in (0., .5, 1.):
            group = [r for r in rows if r['model'] == model and r['kappa'] == kappa]
            summary.append(dict(model=model, kappa=kappa, points=len(group),
                candidate_points=sum(c[0] == model for c in candidates),
                sufficient_points=sum(r['sufficient'] for r in group),
                positive_TS_points=sum(r['TS_derivative'] > 0 for r in group),
                positive_all_fares=sum(min(r[f'price_{m}_derivative'] for m in (1, 2, 3)) > 0 for r in group),
                friction_falls=sum(r['friction_derivative'] < 0 for r in group),
                min_lower_bound=min((r['lower_bound'] for r in group), default=None),
                min_TS_derivative=min((r['TS_derivative'] for r in group), default=None),
                max_derivative_error=max((r['derivative_error'] for r in group), default=None),
                max_jacobian_step_error=max((r['jacobian_step_error'] for r in group), default=None)))
    output.mkdir(parents=True)
    write_csv(output/'local_responses.csv', rows)
    write_csv(output/'coverage.csv', summary)
    (output/'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    manifest = dict(input_sha256={n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in source},
                    summary=summary, failures=len(failures), new_equilibrium_solves=0,
                    interpretation='Local IFT derivative on saved strict response; not a global Nash proof')
    (output/'summary.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
