"""Two-platform actual Nash construction with heterogeneous waiting values.

Global Nash optimality follows from the analytical profit domination argument
in the companion report, never from the finite deviation mesh below.
"""
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, softmax


FORMAL_T = (1., 2., 3., 5., 10., 30., 100., 300., 1000., 3000.)
EXPLORATORY_T = (1., 1.1, 1.5, 2., 3., 5., 10., 30., 100., 1000.)
FD_STEPS = (1e-4, 2e-5, 5e-6)
ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('experiments/aam_pricing/review16_heterogeneity.py',
           'scripts/run_review16_heterogeneity.py', 'tests/test_review16_heterogeneity.py')
PROTECTED = ('submission_packages/TRA_V290_LaTeX_Project/MAIN.tex',
             'experiments/aam_pricing/review14_assumptions.py',
             'scripts/run_review14_assumptions.py', 'tests/test_review14_assumptions.py')


@dataclass(frozen=True)
class Case:
    mass: tuple
    group_effect: tuple
    quality: tuple
    waiting: tuple
    resource: tuple = (1., 10.)
    cost: tuple = (5., 5.)
    outside: tuple = (0., 0.)
    lam: float = 1.
    mu: float = 1.
    capacity: float = 100.
    fare_bounds: tuple = (0., 120.)
    kappa: float = 0.
    p0: float = 0.

    def __post_init__(self):
        for name in ('mass', 'waiting', 'resource'):
            v = np.asarray(getattr(self, name))
            if v.shape != (2,) or not np.all(np.isfinite(v)) or np.min(v) <= 0:
                raise ValueError(name+' must contain two finite positive values')
        for name in ('group_effect', 'quality', 'cost', 'outside'):
            v = np.asarray(getattr(self, name))
            if v.shape != (2,) or not np.all(np.isfinite(v)):
                raise ValueError(name+' must contain two finite values')
        if not all(np.isfinite(z) and z > 0 for z in (self.lam, self.mu, self.capacity)):
            raise ValueError('lam, mu and capacity must be finite positive values')
        if (len(self.fare_bounds) != 2 or not np.all(np.isfinite(self.fare_bounds))
                or self.fare_bounds[0] >= self.fare_bounds[1]):
            raise ValueError('finite increasing fare bounds required')


def family(t, exploratory=False):
    """Only b=(1,t) varies inside either fully specified family."""
    x, a = np.array([.02, .98]), np.array([1., 10.])
    s = (x/a)/sum(x/a)
    p = 5+1/(1-x)
    log_z = 0. if exploratory else -5.
    return Case(mass=(10., 1000.) if exploratory else (11.7, 1000.),
                group_effect=(20., 0.) if exploratory else (30., 0.),
                quality=tuple(p+np.log(s)+log_z), waiting=(1., float(t)))


def _probabilities(case, prices, wait):
    u = (np.array(case.group_effect)[:, None]+np.array(case.quality)[None, :]
         -case.lam*np.asarray(prices)[None, :]
         -np.array(case.waiting)[:, None]*wait)/case.mu
    logits = np.column_stack((np.array(case.outside)/case.mu, u))
    return softmax(logits, axis=1), logits


def response(case, prices):
    """Unique actual complementary response; quality is constant in utilization."""
    prices = np.asarray(prices, dtype=float)
    if prices.shape != (2,) or not np.all(np.isfinite(prices)):
        raise ValueError('two finite fares required')
    mass, a = np.array(case.mass), np.array(case.resource)

    def resource_residual(w):
        probabilities, _ = _probabilities(case, prices, w)
        return float(a@(mass@probabilities[:, 1:])-case.capacity)

    at_zero = resource_residual(0.)
    if at_zero > 0:
        hi = 1.
        for _ in range(100):
            if resource_residual(hi) < 0:
                break
            hi *= 2
        else:
            raise ValueError('waiting bracket exhausted')
        wait = brentq(resource_residual, 0., hi, xtol=5e-15, rtol=1e-14)
        regime = 'positive_wait'
    else:
        wait, regime = 0., 'zero_wait'
    probabilities, logits = _probabilities(case, prices, wait)
    y = mass@probabilities[:, 1:]
    theta = probabilities[:, 1:].sum(axis=1)
    return dict(y=y, theta=theta, probabilities=probabilities[:, 1:],
                outside_probabilities=probabilities[:, 0], wait=wait,
                regime=regime, R=float(a@y),
                CS=float(mass@(case.mu/case.lam*logsumexp(logits, axis=1))),
                group_CS=mass*case.mu/case.lam*logsumexp(logits, axis=1),
                resource_residual=float(a@y-case.capacity), zero_wait_excess=at_zero)


def equilibrium(case, tau):
    """Solve monotone two-share equation, then the partial scalar wait root."""
    a, c, quality = map(np.array, (case.resource, case.cost, case.quality))
    varpi = case.lam/case.mu
    rhs = (quality-case.lam*c)/case.mu+np.log(a)-varpi*a*tau

    def f(x):
        return np.log(x)+1/(1-x)

    x1 = brentq(lambda z: f(z)-f(1-z)-(rhs[0]-rhs[1]),
                1e-10, 1-1e-10, xtol=5e-16, rtol=1e-14)
    x = np.array([x1, 1-x1])
    prices = c+a*tau+1/(varpi*(1-x))
    if np.any(prices <= case.fare_bounds[0]) or np.any(prices >= case.fare_bounds[1]):
        raise ValueError('candidate fares must be interior')
    r = response(case, prices)
    if r['regime'] != 'positive_wait':
        raise ValueError('construction requires positive waiting')
    # Values come from original probabilities, derivatives from the share equations.
    y = r['y']
    omega = 1/(1/x+1/(1-x)**2)
    abar = omega@a/omega.sum()
    dx = varpi*omega*(abar-a)
    dp = a+dx/(varpi*(1-x)**2)
    dy = case.capacity*dx/a
    Y, dY = float(sum(y)), float(sum(dy))
    mass, b, theta = map(np.array, (case.mass, case.waiting, r['theta']))
    L = float(mass@(theta*(1-theta)))
    B = float(mass@(b*theta*(1-theta)))
    H = float(mass@(b*theta))
    avg_dp = float(y@dp/Y)
    dw = -(case.mu*dY+case.lam*L*avg_dp)/B
    margin = prices-c-a*tau
    profit = margin*y
    revenue = tau*float(a@y)
    ground = case.kappa*case.p0*(sum(mass)-Y)
    dCS = -float(y@dp)-H/case.lam*dw
    dtheta = theta*(1-theta)/case.mu*(-case.lam*avg_dp-b*dw)
    dgroup_CS = -mass*theta*(avg_dp+b/case.lam*dw)
    dprofit = (dp-a)*y+margin*dy
    drevenue = float(a@y+tau*(a@dy))
    dground = -case.kappa*case.p0*dY
    allocation = float((prices-c)@dy)
    relief = -H/case.lam*dw
    dTS = relief+allocation+dground
    eta = x/(1-x+x*x)
    Dcoef = float(1+sum(eta*(abar/a-1)))
    common_formula = case.capacity*Dcoef+(Y/(varpi*L)-case.kappa*case.p0)*dY
    result = dict(r, p=prices, x=x, dx=dx, dp=dp, dy=dy, Y=Y, dY=dY,
                  dwait=dw, L=L, B=B, H=H, avg_dp=avg_dp,
                  profit=profit, revenue=revenue, ground=ground,
                  TS=r['CS']+sum(profit)+revenue+ground,
                  dCS=dCS, dprofit=dprofit, drevenue=drevenue, dground=dground,
                  dtheta=dtheta, dgroup_CS=dgroup_CS,
                  dTS=dTS, waiting_relief=relief, allocation=allocation,
                  common_formula=common_formula,
                  common_formula_error=abs(common_formula-dTS),
                  foc=y*(1-varpi*(1-x)*margin),
                  share_identity_error=float(max(abs(y-case.capacity*x/a))),
                  pairwise_dY=float(varpi*case.capacity/omega.sum()*omega[0]*omega[1]
                                    *(a[0]-a[1])**2/(a[0]*a[1])))
    return _plain(result)


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def departures():
    """Exact conditional allocation diagnostics, explicitly not Nash responses."""
    mass, a = np.array([100., 100.]), np.array([1., 2.])
    group, q = np.array([0., .7]), np.array([1., .4])
    p, b, wait, mu, lam = np.array([2., 4.]), np.array([.3, 1.]), .5, 1., .8
    rows = []
    designs = [('control', 0.)]+[(kind, eps) for kind in ('fare', 'interaction')
                               for eps in (.05, .25, .75)]
    for kind, eps in designs:
        lambdas = lam*np.array([1-eps, 1+eps]) if kind == 'fare' else np.full(2, lam)
        interaction = eps*np.array([[1., -1.], [-1., 1.]]) if kind == 'interaction' else np.zeros((2, 2))

        def allocation(prices):
            u = group[:, None]+q+interaction-lambdas[:, None]*prices-b[:, None]*wait
            prob = softmax(np.column_stack((np.zeros(2), u/mu)), axis=1)[:, 1:]
            conditional = softmax((q+interaction-lambdas[:, None]*prices)/mu, axis=1)
            y = mass@prob
            R = float(a@y)
            return prob, conditional, y, R, a*y/R

        prob, cond, y, R, x = allocation(p)
        ref = a*cond[0]/(a@cond[0])
        dprob = -(lambdas/mu)[:, None]*prob*(np.array([1., 0.])-prob[:, :1])
        dy = mass@dprob
        exact_dx = a*dy/R-x*(a@dy)/R
        h = 1e-5
        offset = np.array([h, 0.])
        fd = (allocation(p+offset)[4]-allocation(p-offset)[4])/(2*h)
        rows.append(_plain(dict(kind=kind, epsilon=eps,
            primitives=dict(mass=mass, resource=a, group=group, quality=q, fares=p,
                            waiting=b, wait=wait, mu=mu, lambdas=lambdas, interaction=interaction),
            conditional_shares=cond, conditional_group_gap=float(max(abs(cond[0]-cond[1]))),
            resource_identity_error=float(max(abs(y-R*ref/a))),
            actual_resource_share_derivative=exact_dx,
            auxiliary_own_derivative=-lam/mu*x[0]*(1-x[0]),
            own_share_derivative_error=abs(exact_dx[0]+lam/mu*x[0]*(1-x[0])),
            share_fd_error=float(max(abs(fd-exact_dx))),
            actual_nash_claim=False)))
    return rows


def finite_differences(case, tau=0.):
    """Original logsum and every transfer component, including one-sided tau=0."""
    r = equilibrium(case, tau)
    fields = [('x', 'dx'), ('p', 'dp'), ('y', 'dy'), ('wait', 'dwait'),
              ('CS', 'dCS'), ('profit', 'dprofit'), ('revenue', 'drevenue'),
              ('ground', 'dground'), ('TS', 'dTS'), ('group_CS', 'dgroup_CS'),
              ('theta', 'dtheta')]
    rows = []
    for h in FD_STEPS:
        left, right = equilibrium(case, tau-h), equilibrium(case, tau+h)
        twice = equilibrium(case, tau+2*h)
        estimates = {}
        for key, derivative in fields:
            central = (np.asarray(right[key])-np.asarray(left[key]))/(2*h)
            forward = (-3*np.asarray(r[key])+4*np.asarray(right[key])
                       -np.asarray(twice[key]))/(2*h)
            estimates[key] = dict(analytic=r[derivative], central=central, forward=forward,
                central_error=float(np.max(abs(central-r[derivative]))),
                forward_error=float(np.max(abs(forward-r[derivative]))))
        rows.append(_plain(dict(step=h, estimates=estimates)))
    return rows


def deviation_mesh(case, tau=0., grid_points=121):
    """Finite global-domain corroboration, NOT a global Nash proof."""
    r = equilibrium(case, tau)
    rows = []
    a, c = np.array(case.resource), np.array(case.cost)
    for j in range(2):
        grid = np.unique(np.r_[np.linspace(*case.fare_bounds, grid_points),
                               r['p'][j], c[j]+a[j]*tau])
        for fare in grid:
            p = np.array(r['p'])
            p[j] = fare
            d = response(case, p)
            x = softmax((np.array(case.quality)-case.lam*p)/case.mu+np.log(a))
            margin = fare-c[j]-a[j]*tau
            actual = margin*d['y'][j]
            auxiliary = margin*case.capacity*x[j]/a[j]
            # Air-only log normalization stays finite even when participation underflows.
            _, logits = _probabilities(case, p, d['wait'])
            conditional = softmax(logits[:, 1:], axis=1)
            rows.append(_plain(dict(platform=j, fare=fare, regime=d['regime'],
                wait=d['wait'], R=d['R'], y=d['y'], theta=d['theta'],
                probabilities=d['probabilities'], x=x, margin=margin,
                actual_profit=actual, auxiliary_profit=auxiliary,
                incumbent_profit=r['profit'][j], improvement=actual-r['profit'][j],
                domination_excess=max(0., actual-auxiliary) if margin >= 0 else None,
                resource_share_identity_error=float(max(abs(d['y']-d['R']*x/a))),
                conditional_group_gap=float(max(abs(conditional[0]-conditional[1]))))))
    return rows


def _calculate(grid_points):
    exploratory = [dict(t=t, case=asdict(family(t, True)),
                        result=equilibrium(family(t, True), 0.),
                        negative_TS_found=equilibrium(family(t, True), 0.)['dTS'] < 0)
                   for t in EXPLORATORY_T]
    sweep, deviations = [], []
    for t in FORMAL_T:
        case = family(t)
        r = equilibrium(case, 0.)
        fd = finite_differences(case)
        mesh = deviation_mesh(case, grid_points=grid_points)
        sweep.append(dict(t=t, case=asdict(case), tau=0., result=r,
                          finite_differences=fd,
                          finite_mesh_summary=dict(points=len(mesh),
                              positive_wait=sum(z['regime'] == 'positive_wait' for z in mesh),
                              zero_wait=sum(z['regime'] == 'zero_wait' for z in mesh),
                              maximum_improvement=max(z['improvement'] for z in mesh))))
        deviations.append(dict(t=t, rows=mesh))
    component_errors = [max(v['central_error'], v['forward_error']) for z in sweep
                        for row in z['finite_differences'] for v in row['estimates'].values()]
    all_deviations = [r for z in deviations for r in z['rows']]
    summary = dict(settings=len(sweep), positive_TS_settings=sum(z['result']['dTS'] > 0 for z in sweep),
        negative_TS_settings=sum(z['result']['dTS'] < 0 for z in sweep),
        exploratory_settings=len(exploratory),
        exploratory_negative_TS_settings=sum(z['negative_TS_found'] for z in exploratory),
        min_passenger_derivative=min(z['result']['dY'] for z in sweep),
        max_wait_derivative=max(z['result']['dwait'] for z in sweep),
        direct_resource_residual=max(abs(z['result']['resource_residual']) for z in sweep),
        direct_share_error=max(z['result']['share_identity_error'] for z in sweep),
        exact_foc_residual=max(max(abs(np.array(z['result']['foc']))) for z in sweep),
        fd_max_absolute_error=max(component_errors),
        deviation_points=len(all_deviations),
        deviation_positive_improvement=max(0., max(z['improvement'] for z in all_deviations)),
        deviation_domination_excess=max(z['domination_excess'] or 0. for z in all_deviations),
        deviation_share_identity_error=max(z['resource_share_identity_error'] for z in all_deviations),
        deviation_regimes=sorted(set(z['regime'] for z in all_deviations)),
        global_nash_basis='analytical_profit_domination_not_finite_mesh',
        general_heterogeneous_actual_nash_claim=False)
    if not (summary['positive_TS_settings'] and summary['negative_TS_settings']
            and summary['min_passenger_derivative'] > 0 and summary['max_wait_derivative'] < 0
            and summary['direct_resource_residual'] < 1e-7 and summary['exact_foc_residual'] < 1e-8
            and summary['fd_max_absolute_error'] < 1e-4
            and summary['deviation_positive_improvement'] < 1e-7
            and summary['deviation_domination_excess'] < 1e-7
            and summary['deviation_regimes'] == ['positive_wait', 'zero_wait']):
        raise ValueError('Frozen batch validation failed: '+json.dumps(summary))
    return _plain(dict(exploratory=exploratory, sweep=sweep, deviations=deviations,
                       departures=departures(), summary=summary))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, obj):
    path.write_text(json.dumps(_plain(obj), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write(folder, grid_points=121):
    """Freeze before running; never overwrite an existing experiment directory."""
    if not isinstance(grid_points, int) or grid_points < 3:
        raise ValueError('at least three grid points required')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    frozen = dict(formal_t=FORMAL_T, exploratory_t=EXPLORATORY_T, tau=0., fd_steps=FD_STEPS,
        cases=[asdict(family(t)) for t in FORMAL_T],
        exploratory_cases=[asdict(family(t, True)) for t in EXPLORATORY_T],
        grid_points=grid_points, python=sys.executable,
        source_sha256={name: _sha(ROOT/name) for name in SOURCES},
        protected_before_sha256={name: _sha(ROOT/name) for name in PROTECTED},
        design='two_group_two_platform_heterogeneous_positive_b_fixed_other_primitives',
        actual_nash_proof='analytical; finite mesh is corroboration only')
    _save(folder/'frozen_design.json', frozen)
    try:
        batch = _calculate(grid_points)
    except Exception as error:
        _save(folder/'failed_attempt.json', dict(error_type=type(error).__name__, error=str(error)))
        raise
    for name, obj in batch.items():
        _save(folder/(name+'.json'), obj)
    protected_after = {name: _sha(ROOT/name) for name in PROTECTED}
    _save(folder/'scope_guard.json', dict(before=frozen['protected_before_sha256'],
          after=protected_after, observed_unchanged=protected_after == frozen['protected_before_sha256']))
    with (folder/'sweep.csv').open('w', newline='', encoding='utf-8') as handle:
        columns = ['t', 'wait', 'dwait', 'dY', 'L', 'B', 'H', 'waiting_relief', 'allocation', 'dTS']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in batch['sweep']:
            writer.writerow(dict(t=row['t'], **{k: row['result'][k] for k in columns[1:]}))
    _save(folder/'manifest.json', dict(files={p.name: _sha(p) for p in sorted(folder.iterdir())
        if p.is_file()}, source_sha256=frozen['source_sha256']))
    return batch['summary']


def replay(folder):
    """Hash verification followed by a full fresh deterministic numerical replay."""
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if _sha(folder/name) != digest:
            raise ValueError('Artifact hash changed: '+name)
    for name, digest in manifest['source_sha256'].items():
        if _sha(ROOT/name) != digest:
            raise ValueError('Source hash changed: '+name)
    frozen = json.loads((folder/'frozen_design.json').read_text(encoding='utf-8'))
    fresh = _calculate(frozen['grid_points'])
    for name, expected in fresh.items():
        saved = json.loads((folder/(name+'.json')).read_text(encoding='utf-8'))
        if saved != expected:
            raise ValueError('Numerical replay changed: '+name)
    return fresh['summary']
