"""Actual common waiting in minutes; finite diagnostics, no global BR claim."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .review_equilibrium import ReviewOutcome, ReviewEngine
from .review_hard_capacity_repair import RepairEngine, smooth_polish, verify_candidate
from .review90_allocation import group_surplus_difference


@dataclass
class WaitingOutcome(ReviewOutcome):
    wait_minutes: float = 0.


class WaitingEngine(RepairEngine):
    def __init__(self, case, budget=None, beta=None):
        super().__init__(case, budget)
        self.beta = np.array(self.time_sensitivity if beta is None else beta, dtype=float)
        if self.beta.shape != self.demand_base.shape or not np.all(np.isfinite(self.beta)) or np.any(self.beta <= 0):
            raise ValueError('Positive finite waiting sensitivity per group required')

    def choice(self, prices, rho, wait_minutes=0.):
        self.budget.choice_calls += 1
        loss = self.base_loss+self.exposure*self.source.reliability_function(rho)
        air = (self.platform_affinity-self.price_sensitivity[:,None]*prices
               -self.time_sensitivity[:,None]*self.time
               -self.reliability_sensitivity[:,None]*loss
               -self.beta[:,None]*wait_minutes)/self.logit_scale
        outside = (self.outside_affinity-self.price_sensitivity*self.case.outside_price
                   -self.time_sensitivity*self.case.outside_time)/self.logit_scale
        den = np.logaddexp(outside, logsumexp(air,axis=1))
        prob = np.exp(air-den[:,None])
        return self.demand_base @ prob, prob, float(np.dot(self.demand_base*self.logit_scale/self.price_sensitivity,den))

    def outcome(self, prices):
        p = np.asarray(prices,dtype=float)
        key = tuple(p)
        if key in self.cache:
            return self.cache[key]
        self.budget.spend('waiting_closed_demand',p)
        def gap(w):
            return float(self.resource @ self.choice(p,1.,w)[0]/self.capacity-1)
        if gap(0.) > 0:
            hi = 1.
            while gap(hi) > 0:
                hi *= 2.
            wait = brentq(gap,0.,hi,xtol=2e-12,rtol=1e-14)
            y,prob,cs = self.choice(p,1.,wait)
            o = WaitingOutcome(y,1.,prob,cs,True,0.,abs(gap(wait)),wait)
        else:
            rho = brentq(lambda r: float(self.resource @ self.choice(p,r)[0]/self.capacity-r),
                         0.,1.,xtol=2e-13,rtol=1e-14)
            y,prob,cs = self.choice(p,rho)
            o = WaitingOutcome(y,rho,prob,cs,False,0.,abs(float(self.resource @ y/self.capacity-rho)),0.)
        self.cache[key] = o
        return o

    def jacobian(self, prices, outcome=None):
        o = self.outcome(prices) if outcome is None else outcome
        if not o.binding:
            return ReviewEngine.jacobian(self,prices,o)
        p = o.prob
        weights = self.demand_base*self.price_sensitivity/self.logit_scale
        direct = np.einsum('g,gm,gj->mj',weights,p,p)-np.diag(weights @ p)
        wait_derivative = -(self.demand_base*self.beta/self.logit_scale*(1-p.sum(axis=1))) @ p
        wait_price = -(self.resource @ direct)/float(self.resource @ wait_derivative)
        return direct+np.outer(wait_derivative,wait_price)


def group_comparison(engine, before, after):
    lo,hi = before.prob.sum(axis=1),after.prob.sum(axis=1)
    delta = group_surplus_difference(engine.demand_base,engine.price_sensitivity,lo,hi,engine.logit_scale)
    outside = (engine.outside_affinity-engine.price_sensitivity*engine.case.outside_price
               -engine.time_sensitivity*engine.case.outside_time)/engine.logit_scale
    # Fixed outside utility permits a share-based reconstruction independent
    # of the direct utility calculation in group_logsum.
    levels0 = engine.demand_base*engine.logit_scale/engine.price_sensitivity*(outside-np.log1p(-lo))
    levels1 = engine.demand_base*engine.logit_scale/engine.price_sensitivity*(outside-np.log1p(-hi))
    return [dict(group=g+1,group_mass=float(engine.demand_base[g]),
                 money_sensitivity=float(engine.price_sensitivity[g]),
                 time_sensitivity=float(engine.time_sensitivity[g]),
                 value_of_time_per_minute=float(engine.time_sensitivity[g]/engine.price_sensitivity[g]),
                 service_probability_NC=float(lo[g]),service_probability_SD=float(hi[g]),
                 service_probability_change=float(hi[g]-lo[g]),
                 served_NC=float(engine.demand_base[g]*lo[g]),served_SD=float(engine.demand_base[g]*hi[g]),
                 aggregate_cs_change=float(delta[g]),perperson_net_cs_change=float(delta[g]/engine.demand_base[g]),
                 reconstructed_cs_NC=float(levels0[g]),reconstructed_cs_SD=float(levels1[g]),
                 reconstruction_error=float(levels1[g]-levels0[g]-delta[g])) for g in range(3)]


def group_logsum(engine, prices, outcome):
    loss = engine.base_loss+engine.exposure*engine.source.reliability_function(outcome.rho)
    air = (engine.platform_affinity-engine.price_sensitivity[:,None]*np.asarray(prices)
           -engine.time_sensitivity[:,None]*engine.time
           -engine.reliability_sensitivity[:,None]*loss)/engine.logit_scale
    if isinstance(engine, WaitingEngine):
        air -= engine.beta[:,None]*outcome.wait_minutes/engine.logit_scale
    else:
        air += outcome.logphi
    outside = (engine.outside_affinity-engine.price_sensitivity*engine.case.outside_price
               -engine.time_sensitivity*engine.case.outside_time)/engine.logit_scale
    return engine.demand_base*engine.logit_scale/engine.price_sensitivity*np.logaddexp(outside,logsumexp(air,axis=1))


def profile_design():
    pilot = [(d,k,p) for d,k in [(.85,1.12),(1.35,.90),(1.70,.70)] for p in [1.70,2.20,2.80]]
    full = pilot+[(d,k,p) for d,k in [(1.,.80),(1.55,1.),(1.15,.70)] for p in [1.70,2.20,2.80]]
    return pilot,full


def final_polish(engine, prices, charge):
    return verify_candidate(engine,smooth_polish(engine,prices,charge),charge)
