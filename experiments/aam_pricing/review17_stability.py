"""Conditional local-to-global bounds, exact reference, no equilibrium search.

Envelope values are inputs requiring independent whole-domain verification.
The arithmetic example is not a computed primitive heterogeneity radius.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

OUTPUT = 'results/trb_review17_stability_v1'


def design():
    return dict(mode='pure_theory_with_exact_reference',
                primitive_perturbation_points=[], maximum_points=12,
                compute_budget_seconds=600, new_Nash_solves=0,
                scope='conditional_local_global_continuation_not_general_mixed_logit')


def reference():
    x, a = [F(1, 2)]*2, [F(1), F(2)]
    y = [v/ai for v, ai in zip(x, a)]
    eta = [v/(1-v+v*v) for v in x]
    omega = [et*(1-v)**2 for et, v in zip(eta, x)]
    abar = sum(w*ai for w, ai in zip(omega, a))/sum(omega)
    dx = [w*(abar-ai) for w, ai in zip(omega, a)]
    dy = [v/ai for v, ai in zip(dx, a)]
    dp = [(1-et)*ai+et*abar for et, ai in zip(eta, a)]
    Y, L, dY = sum(y), F(15, 32), sum(dy)
    mw = sum(v*p for v, p in zip(y, dp))/Y+dY/L
    J = [[-y[i] if i == j else y[i]*x[i]*x[j]/(1-x[i])
          for j in range(2)] for i in range(2)]
    det = J[0][0]*J[1][1]-J[0][1]*J[1][0]
    inverse = [[J[1][1]/det, -J[0][1]/det],
               [-J[1][0]/det, J[0][0]/det]]
    radius, curvature = F(1, 4), F(5, 36)
    return dict(prices=[F(8), F(9)], costs=[F(5), F(5)], charge=F(1),
                resources=a, mass=[F(1), F(1)], quality=['9+ln(2)', '10'],
                outside=['ln(5)', 'ln(5)'], fare_box=[F(5), F(44)],
                lambda_mu_b=[F(1), F(1), F(1)], wait=F(1), y=y, x=x,
                profit=[2*v for v in y], Y=Y, L=L, Ydot=dY, pdot=dp,
                minus_wdot=mw, Wdot_air=Y*mw+3*dy[0]+4*dy[1],
                J=J, inverse=inverse,
                inverse_norm=max(sum(abs(v) for v in row) for row in inverse),
                local_radius=radius, curvature_lower=curvature,
                curvature_absolute_upper=F(1),
                remote_gap_lower=curvature*radius**2/2,
                closure_wait_slope_absolute=F(5, 8),
                reference_exact_global_Nash=True)


def example_envelopes():
    r = reference()
    return dict(r=r['local_radius'], s=F(1, 100), q=F(1, 2),
                residual=F(1, 1000), fare_slack=F(3), wait_lower=F(1, 2),
                curvature=r['curvature_lower'],
                curvature_drop=r['curvature_lower']/2,
                C=r['curvature_absolute_upper'], gap=r['remote_gap_lower'],
                payoff_error=r['remote_gap_lower']/8,
                Ydot_error=F(1, 1000), wait_slope_error=F(1, 100),
                Wdot_air_error=F(1, 100), eta_upper=F(1))


def check_envelopes(envelopes):
    e = {key: F(value) for key, value in envelopes.items()}
    if any(v < 0 for v in e.values()):
        raise ValueError('All supplied error bounds and margins must be nonnegative')
    local = (0 < e['s'] < e['r'] < e['fare_slack']
             and e['wait_lower'] > 0 and e['q'] < 1)
    residual_ok = e['residual'] <= (1-e['q'])*e['s']
    stationary = local and residual_ok
    concave = e['curvature'] > e['curvature_drop']
    margin = e['gap']-e['C']*e['s']**2/2-2*e['payoff_error']
    global_nash = stationary and concave and margin > 0 and e['C'] > 0
    r = reference()
    yd = r['Ydot']-e['Ydot_error']
    wm = r['minus_wdot']-e['wait_slope_error']
    welfare = (r['Wdot_air']-e['Wdot_air_error']
               -e['eta_upper']*(r['Ydot']+e['Ydot_error']))
    return dict(stationary_root=stationary, global_Nash=global_nash,
                signs_preserved=global_nash and yd > 0 and wm > 0 and welfare > 0,
                global_margin=margin, Ydot_lower=yd, minus_wdot_lower=wm,
                Wdot_lower=welfare,
                evidence='conditional_bounds_not_a_verified_primitive_parameter_box')


def tie_payoff(q, perturbation):
    q, perturbation = F(q), F(perturbation)
    return -q*q*(q-1)**2+perturbation*q**3


def tie_local_derivatives(perturbation):
    F(perturbation)
    return F(0), F(-2)


def serialized(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: serialized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialized(v) for v in value]
    return value


def bundle():
    e = example_envelopes()
    positive = check_envelopes(e)
    negative = check_envelopes(dict(e, payoff_error=F(1, 100)))
    assert positive['global_Nash'] and positive['signs_preserved']
    assert negative['stationary_root'] and not negative['global_Nash']
    return serialized(dict(design=design(), exact_reference=reference(),
                           conditional_envelopes=e, conditional_conclusion=positive,
                           counterchecks=dict(local_only_implies_global=False,
                                              failed_remote_condition=negative,
                                              tie_perturbation='1/1000000000',
                                              tie_remote_gain=tie_payoff(1, F(1, 10**9)),
                                              tie_local_derivatives=tie_local_derivatives(0)),
                           verification=dict(valid=True, new_Nash_solves=0,
                                             primitive_radius_certified=False,
                                             finite_market_experiments=0)))


def run(root, destination=None):
    root = Path(root).resolve()
    owned = (root/OUTPUT).resolve()
    destination = Path(destination or owned).resolve()
    if not destination.is_relative_to(owned):
        raise ValueError('Output outside owned result tree')
    data = bundle()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/'theory_checks.json'
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    sources = ['experiments/aam_pricing/review17_stability.py',
               'tests/test_review17_stability.py', 'scripts/run_review17_stability.py',
               'docs/TRB_REVIEW17_STABILITY_20261019.md']
    manifest = dict(source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest()
                                   for p in sources if (root/p).exists()},
                    result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    new_Nash_solves=0)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return data['verification']


def replay(root, destination=None):
    root = Path(root).resolve()
    owned = (root/OUTPUT).resolve()
    destination = Path(destination or owned).resolve()
    if not destination.is_relative_to(owned):
        raise ValueError('Output outside owned result tree')
    path = destination/'theory_checks.json'
    if json.loads(path.read_text(encoding='utf-8')) != bundle():
        raise ValueError('Arithmetic replay mismatch')
    m = json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != m['result_sha256']:
        raise ValueError('Result hash mismatch')
    for p, h in m['source_sha256'].items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest() != h:
            raise ValueError('Source hash mismatch: '+p)
    return dict(valid=True, new_Nash_solves=0, primitive_radius_certified=False)
