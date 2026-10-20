"""Complete NC baseline and all-pairs guarantees for the exact two-platform game.

Reuses only the historical outward arithmetic and proved post-release roots.
No optimizer, saved-candidate range or fare-grid diagnostic enters the result.
The continuum completeness argument is in docs/review18_selection_20261020.md.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from .review16_selection_range import (
    I, UP, NEAR, CHARGES, exact, midpoint, capacity_state,
    continuum_bounds, root_brackets,
)

OUTPUT = 'results/review18_selection_v1'
ROOT_WIDTH = Decimal('1e-40')
SOURCES = (
    'experiments/aam_pricing/review18_selection.py',
    'scripts/run_review18_selection.py',
    'tests/test_review18_selection.py',
    'docs/review18_selection_20261020.md',
    'experiments/aam_pricing/review16_selection_range.py',
)


def share_residual(r, charge):
    r = r if isinstance(r, I) else I(r)
    return (r/(1-r)).log()-I(charge)-I(1)/r+I(1)/(1-r)+1


def share_root(charge):
    """Unique auxiliary resource share, with signed endpoints, for 0 <= tau <= 1."""
    t = exact(charge)
    if not 0 <= t <= 1:
        raise ValueError('Prerelease proof requires charge in [0,1]')
    if t == 1:
        return I('0.5')
    lo, hi = Decimal('0.4'), Decimal('0.5')
    if not (share_residual(lo, t).hi < 0 < share_residual(hi, t).lo):
        raise ArithmeticError('Share endpoint signs not established')
    while UP.subtract(hi, lo) > ROOT_WIDTH:
        mid = midpoint(I(lo, hi))
        f = share_residual(mid, t)
        if f.hi < 0:
            lo = mid
        elif f.lo > 0:
            hi = mid
        else:
            raise ArithmeticError('Increase arithmetic precision')
    return I(lo, hi)


def unique_prerelease(charge):
    """Enclose the full singleton actual Nash correspondence before release."""
    t = exact(charge)
    r = share_root(t)
    p1, p2 = 5+I(t)+I(1)/(1-r), 5+2*I(t)+I(1)/r
    z1, z2 = 2*(8-p1).exp(), (9-p2).exp()
    waiting = ((z1+3*z2)/5).log() if t != 1 else I(0)
    cs = 2*(I(20)/(3-r)).log()
    y1, y2 = r, (1-r)/2
    m1, m2 = p1-5-I(t), p2-5-2*I(t)
    welfare = cs+(p1-5)*y1+(p2-5)*y2
    return dict(charge=str(t), full_actual_Nash_set='singleton', share=r,
                p1=p1, p2=p2, y1=y1, y2=y2, waiting=waiting, cs=cs,
                resource=I(1), profit1=m1*y1, profit2=m2*y2, welfare=welfare,
                auxiliary_residual1=m1*(1-r)-1,
                auxiliary_residual2=m2*r-1)


def gain_comparison(nc_min, nc_max, charged_min, charged_max):
    """Extrema must cover COMPLETE Nash sets; candidates do not suffice."""
    return dict(difference_of_worst_levels=(charged_min-nc_min).serialized(),
                all_pairs_guaranteed_gain=(charged_min-nc_max).serialized(),
                best_possible_gain=(charged_max-nc_min).serialized(),
                baseline_selection_width=(nc_max-nc_min).serialized())


def proof_bounds():
    bounds = dict(
        H_at_two_fifths_tau0=share_residual('0.4', '0'),
        share_residual_derivative_lower=I(12),
        welfare_share_derivative_lower=I(395)/936,
        global_margin1_lower=I('1.9')-I('2.5').log(),
        global_margin2_lower=I('1.8')-(I(5)/3).log(),
        slack_contradiction_excess=2*I('-0.1').exp()+3*(I(7)/15).exp()-5,
        auxiliary_contraction=(I(40).exp())/(1+I(40).exp()),
    )
    positive = ('share_residual_derivative_lower', 'welfare_share_derivative_lower',
                'global_margin1_lower', 'global_margin2_lower', 'slack_contradiction_excess')
    post = continuum_bounds()
    valid = (all(bounds[k].lo > 0 for k in positive)
             and bounds['H_at_two_fifths_tau0'].hi < 0
             and bounds['auxiliary_contraction'].hi < 1 and post['valid'])
    if not valid:
        raise ArithmeticError('Continuum proof constants failed')
    return dict(valid=valid, domain=['0', '1.1'],
                bounds={k: v.serialized() for k, v in bounds.items()},
                post_release_bounds=post,
                proof='docs/review18_selection_20261020.md',
                completeness='singleton on [0,1]; full kink interval on (1,1.1]')


def _serialize_state(state):
    return {k: v.serialized() if isinstance(v, I) else v for k, v in state.items()}


def bundle():
    proof = proof_bounds()
    nc = unique_prerelease('0')
    base = nc['welfare']
    release = capacity_state('0', '0')['welfare']
    rows = [dict(charge='0', worst_welfare=base.serialized(), best_welfare=base.serialized(),
                 difference_of_worst_levels=['0', '0'], all_pairs_guaranteed_gain=['0', '0'],
                 best_possible_gain=['0', '0'], baseline_selection_width=['0', '0'],
                 full_actual_Nash_set='singleton')]
    for charge in CHARGES:
        e = NEAR.subtract(Decimal(charge), Decimal(1))
        low, high = root_brackets(e)
        best = capacity_state(low, e)['welfare']
        worst = capacity_state(high, e)['welfare']
        gains = gain_comparison(base, base, worst, best)
        # Equality follows analytically from singleton NC, not interval subtraction.
        gains['baseline_selection_width'] = ['0', '0']
        rows.append(dict(charge=charge, d_min=low.serialized(), d_max=high.serialized(),
                         worst_welfare=worst.serialized(), best_welfare=best.serialized(),
                         full_actual_Nash_set='singleton' if e == 0 else 'closed_continuum',
                         worst_change_from_release=(worst-release if e else I(0)).serialized(),
                         best_change_from_release=(best-release if e else I(0)).serialized(),
                         **gains))
    # Interval separation proves the decision, rather than sorting rounded midpoints.
    charged = rows[1:]
    valid = (all(Decimal(row['all_pairs_guaranteed_gain'][0]) > 0 for row in charged)
             and all(Decimal(a['worst_welfare'][0]) > Decimal(b['worst_welfare'][1])
                     for a, b in zip(charged, charged[1:]))
             and all(Decimal(a['best_welfare'][1]) < Decimal(b['best_welfare'][0])
                     for a, b in zip(charged, charged[1:])))
    if not valid:
        raise ArithmeticError('Policy ordering not separated by outward intervals')
    return dict(
        menu=['0']+list(CHARGES), nc=_serialize_state(nc), proof=proof, ranges=rows,
        policy=dict(maximin_charges=['1'], all_pairs_guarantee_maximizers=['1'],
                    optimistic_charges=['1.1'],
                    constant_release_fare_maximizers=list(CHARGES),
                    continuum_domain=['0', '1.1'],
                    continuous_maximin_charge='1', continuous_optimistic_charge='1.1'),
        verification=dict(valid=True, new_engine_calls=0, precision=72,
                          nc_root_width=str(ROOT_WIDTH),
                          post_release_root_width='1e-38',
                          scope='complete_actual_Nash_correspondence_of_exact_two_platform_game',
                          heterogeneous_full_correspondence_claim=False,
                          candidate_sets_used=False,
                          source_of_guarantee='continuum_proof_and_outward_arithmetic'))


def _destination(root, destination):
    owned = (Path(root)/OUTPUT).resolve()
    path = Path(destination or owned).resolve()
    if not path.is_relative_to(owned):
        raise ValueError('Output must remain inside '+str(owned))
    return path


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+'\n'


def run(root, destination=None):
    root = Path(root).resolve()
    destination = _destination(root, destination)
    data = bundle()
    destination.mkdir(parents=True, exist_ok=True)
    payload = destination/'selection.json'
    payload.write_text(_json(data), encoding='utf-8')
    manifest = dict(source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest()
                                   for p in SOURCES},
                    result_sha256={'selection.json': hashlib.sha256(payload.read_bytes()).hexdigest()})
    (destination/'manifest.json').write_text(_json(manifest), encoding='utf-8')
    return data['verification']


def replay(root, destination=None):
    root = Path(root).resolve()
    destination = _destination(root, destination)
    manifest = json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise AssertionError('Source hash mismatch: '+name)
    for name, digest in manifest['result_sha256'].items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest() != digest:
            raise AssertionError('Result hash mismatch: '+name)
    fresh = bundle()
    if json.loads((destination/'selection.json').read_text(encoding='utf-8')) != fresh:
        raise AssertionError('Arithmetic replay mismatch')
    return dict(valid=True, replay='full arithmetic and all source/result hashes agree',
                new_engine_calls=0)
