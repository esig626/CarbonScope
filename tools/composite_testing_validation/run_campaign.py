"""Deterministic validation campaign; run from the repository root.

Example: PYTHONPATH=src:. python -m tools.composite_testing_validation.run_campaign
    --label repaired --random 400 --stress 10000

To audit the original implementation, put an exported base commit's src first
on PYTHONPATH. This module never calls a production private solver/enumerator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, localcontext
import json
import hashlib
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.exceptions import FluxEMUError
from fluxemu.testing import (
    CompositeBinaryTestingProblem, CompositeMIDLawFamily, IndependentMIDProductLaw,
    calibrate_composite_score_test, composite_renyi_converse_at_order,
    composite_renyi_score_candidate, exact_finite_composite_minimax,
)
from .oracle import decimal, enumerate_masses, errors, solve, dual_minimax, vertex_minimax


SEED = 20260909
BASE_SHA = '4f457d2afd75ad1bdffdbe0e9eea25fcaf8a9ad4'
VALUE_TOLERANCE = 2e-9
EXACT_ORACLE_TOLERANCE = 5e-10


def problem_from_blocks(null, alternative, totals):
    def family(rows, prefix):
        return CompositeMIDLawFamily(members=tuple(IndependentMIDProductLaw(
            blocks=tuple(MultinomialMIDLaw(n, tuple(float(x) for x in p))
                         for n, p in zip(totals, row, strict=True))) for row in rows),
            member_ids=tuple(f'{prefix}{i}' for i in range(len(rows))))
    return CompositeBinaryTestingProblem(null=family(null, 'P'), alternative=family(alternative, 'Q'))


def categorical(null, alternative, n=1):
    return problem_from_blocks(tuple((p,) for p in null), tuple((q,) for q in alternative), (n,))


def specification(problem):
    return {'totals': problem.null.members[0].block_totals,
            'null': [[b.probabilities for b in m.blocks] for m in problem.null.members],
            'alternative': [[b.probabilities for b in m.blocks] for m in problem.alternative.members]}


class Campaign:
    def __init__(self, label):
        self.label = label
        self.counts = Counter()
        self.refusals = Counter()
        self.refusal_examples = {}
        self.failures = Counter()
        self.examples = {}
        self.metrics = {'max_oracle_gap': 0., 'max_direct_error_discrepancy': 0.,
                        'max_projected_optimality_gap': 0., 'min_converse_gap': None,
                        'max_converse_gap': None}
        self.refusal_curves = {}
        self.extremal_witnesses = {}
        self.source_hashes = {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in tuple(sys.modules.items())
            if name.startswith(('fluxemu.testing.', 'fluxemu.observation.'))
            and getattr(module, '__file__', '').endswith('.py')
        }

    def failure(self, category, case, detail):
        self.failures[category] += 1
        examples = self.examples.setdefault(category, [])
        if len(examples) < 3:
            examples.append({'case': case, **detail})

    def refusal(self, stage, error):
        reason = str(error).split(';')[0].split(':')[0]
        key = stage + ':' + type(error).__name__ + ':' + reason
        self.refusals[key] += 1
        examples = self.refusal_examples.setdefault(key, [])
        if len(examples) < 3 and str(error) not in examples:
            examples.append(str(error))

    def check(self, problem, eps, case, *, group, order=2., score_order=.5, vertices=False, score=True):
        self.counts[group] += 1
        outcomes, p, q = enumerate_masses(problem)
        try:
            oracle = solve(p, q, eps, vertices=vertices)
            self.counts[oracle.method] += 1
        except ArithmeticError as error:
            self.refusal('oracle', error)
            if len(outcomes) <= 4:
                oracle = vertex_minimax(p, q, eps)
                self.counts['vertex_oracle_fallback'] += 1
            else:
                oracle = None
        try:
            lp = exact_finite_composite_minimax(problem, epsilon=eps, max_outcomes=10000)
        except FluxEMUError as error:
            self.counts['lp_refused'] += 1
            self.refusal('lp', error)
            lp = None
        except Exception as error:
            self.failure('unexpected_lp_exception', case, {'exception': repr(error)})
            lp = None
        detail = {'epsilon': eps, **specification(problem)}
        if lp is not None:
            self.counts['lp_accepted'] += 1
            index = {outcome: i for i, outcome in enumerate(lp.outcomes)}
            phi = tuple(lp.rejection_probabilities[index[y]] for y in outcomes)
            alpha, beta = errors(p, q, phi)
            exact_beta = float(max(beta))
            discrepancy = max(abs(float(a) - b) for a, b in zip(alpha + beta, lp.null_type_i_errors + lp.alternative_type_ii_errors, strict=True))
            self.metrics['max_direct_error_discrepancy'] = max(self.metrics['max_direct_error_discrepancy'], discrepancy)
            if any(not math.isfinite(x) or x < 0 or x > 1 for x in phi):
                self.failure('accepted_invalid_phi', case, detail | {'phi': phi})
            if max(alpha) > decimal(eps) * decimal(1 + 5e-9):
                self.failure('accepted_type_i_violation', case, detail | {'alpha': float(max(alpha))})
            if discrepancy > VALUE_TOLERANCE or abs(exact_beta - lp.minimax_type_ii_error) > VALUE_TOLERANCE:
                self.failure('reported_error_mismatch', case, detail | {'direct_beta': exact_beta, 'reported_beta': lp.minimax_type_ii_error})
            if hasattr(lp, 'solver_objective'):
                tolerance = lp.numerical_tolerance
                if (abs(lp.solver_objective - lp.epigraph_variable) > tolerance
                        or abs(lp.epigraph_variable - exact_beta) > tolerance
                        or lp.dual_lower_bound > exact_beta + tolerance
                        or abs(lp.optimality_gap - (lp.minimax_type_ii_error - lp.dual_lower_bound)) > tolerance):
                    self.failure('lp_objective_certificate_mismatch', case, detail)
                else:
                    self.counts['lp_result_certificate_checked'] += 1
            if oracle is not None:
                gap = exact_beta - oracle.beta
                if abs(gap) > EXACT_ORACLE_TOLERANCE and oracle.rejection is None and len(outcomes) <= 4:
                    oracle = vertex_minimax(p, q, eps)
                    self.counts['vertex_oracle_fallback'] += 1
                    gap = exact_beta - oracle.beta
                self.metrics['max_oracle_gap'] = max(self.metrics['max_oracle_gap'], abs(gap))
                allowed = EXACT_ORACLE_TOLERANCE if oracle.rejection is not None else VALUE_TOLERANCE
                if abs(gap) > allowed:
                    self.failure('accepted_optimality_uncertified', case, detail | {'beta': exact_beta, 'oracle': asdict(oracle)})
                else:
                    self.counts['independently_optimality_certified'] += 1
                if vertices:
                    dual = dual_minimax(p, q, eps)
                    if abs(dual.beta - oracle.beta) > VALUE_TOLERANCE:
                        self.failure('independent_oracles_disagree', case, detail | {'vertex': oracle.beta, 'dual': dual.beta})
            else:
                self.failure('accepted_lp_without_oracle_certificate', case, detail)
        try:
            converse = composite_renyi_converse_at_order(problem, epsilon=eps, order=order)
        except FluxEMUError as error:
            self.refusal('converse', error)
            converse = None
        except Exception as error:
            self.failure('unexpected_converse_exception', case, detail | {'exception': repr(error)})
            converse = None
        if converse is not None and oracle is not None and (lp is not None or oracle.rejection is not None):
            self.counts['converse_compared'] += 1
            # A dual lower bound alone does not certify the optimum. Use its
            # agreement with the primal, or the analytical/vertex optimum.
            if lp is not None and abs(lp.minimax_type_ii_error - oracle.beta) <= VALUE_TOLERANCE or oracle.rejection is not None:
                gap = oracle.beta - converse.type_ii_lower_bound
                for key, fn in (('min_converse_gap', min), ('max_converse_gap', max)):
                    old = self.metrics[key]
                    self.metrics[key] = gap if old is None else fn(old, gap)
                    if old is None or self.metrics[key] != old:
                        self.extremal_witnesses[key] = detail | {'case': case, 'gap': gap, 'order': order}
                if gap < -VALUE_TOLERANCE:
                    self.failure('false_converse', case, detail | {'order': order, 'converse': converse.type_ii_lower_bound, 'oracle': oracle.beta})
        if not score:
            return lp is not None
        try:
            candidate = composite_renyi_score_candidate(problem, order=score_order)
            self.counts['candidate_verified' if candidate.uniform_moment_bounds_verified else 'candidate_refused'] += 1
            if candidate.uniform_moment_bounds_verified:
                selected_p = p[candidate.null_member_index]
                selected_q = q[candidate.alternative_member_index]
                with localcontext() as context:
                    context.prec = 70
                    lam = decimal(score_order)
                    z = sum((pi ** (1 - lam) * qi ** lam for pi, qi in zip(selected_p, selected_q, strict=True) if pi and qi), Decimal(0))
                    for family, exponent in ((p, lam), (q, lam - 1)):
                        for row in family:
                            moment = Decimal(0)
                            for weight, pi, qi in zip(row, selected_p, selected_q, strict=True):
                                if not weight:
                                    continue
                                if not pi and not qi:
                                    moment = Decimal('Infinity'); break
                                if not pi:
                                    term = Decimal('Infinity') if exponent > 0 else Decimal(0)
                                elif not qi:
                                    term = Decimal(0) if exponent > 0 else Decimal('Infinity')
                                else:
                                    term = (qi / pi) ** exponent
                                moment += weight * term
                            if moment > z + Decimal('1e-12'):
                                self.failure('false_uniform_moment_certification', case, detail | {'score_order': score_order, 'excess': float(moment - z)})
                                break
            calibrated = calibrate_composite_score_test(candidate, epsilon=eps, max_outcomes=10000)
            self.counts['projected_accepted'] += 1
            index = {y: i for i, y in enumerate(calibrated.outcomes)}
            alpha, beta = errors(p, q, tuple(calibrated.rejection_probabilities[index[y]] for y in outcomes))
            if max(alpha) > decimal(eps) * decimal(1 + 5e-9):
                self.failure('projected_type_i_violation', case, detail | {'alpha': float(max(alpha))})
            if abs(float(max(beta)) - calibrated.worst_type_ii_error) > VALUE_TOLERANCE:
                self.failure('projected_error_mismatch', case, detail)
            if lp is not None:
                gap = float(max(beta)) - lp.minimax_type_ii_error
                if gap > self.metrics['max_projected_optimality_gap']:
                    self.extremal_witnesses['max_projected_optimality_gap'] = detail | {
                        'case': case, 'gap': gap, 'score_order': score_order,
                        'minimax_beta': lp.minimax_type_ii_error, 'projected_beta': float(max(beta)),
                        'uniform_moments_verified': candidate.uniform_moment_bounds_verified}
                self.metrics['max_projected_optimality_gap'] = max(self.metrics['max_projected_optimality_gap'], gap)
                if gap < -VALUE_TOLERANCE:
                    self.failure('projected_below_minimax', case, detail | {'gap': gap})
                if converse is not None:
                    self.counts['central_invariant_compared'] += 1
                    if converse.type_ii_lower_bound > lp.minimax_type_ii_error + VALUE_TOLERANCE:
                        self.failure('central_invariant_converse', case, detail)
        except FluxEMUError as error:
            self.refusal('score', error)
        except Exception as error:
            self.failure('unexpected_score_exception', case, detail | {'exception': repr(error)})
        return lp is not None

    def summary(self):
        return {'label': self.label, 'seed': SEED, 'base_sha': BASE_SHA,
                'executed_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'audited_source_sha256': self.source_hashes,
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'value_tolerance': VALUE_TOLERANCE, 'exact_oracle_tolerance': EXACT_ORACLE_TOLERANCE,
                'counts': dict(self.counts),
                'failure_counts': dict(self.failures), 'failure_examples': self.examples,
                'refusal_counts': dict(self.refusals), 'refusal_examples': self.refusal_examples,
                'metrics': self.metrics,
                'extremal_witnesses': self.extremal_witnesses,
                'count_law_refusal_curves': self.refusal_curves,
                'affine_control': 'New explicit finite grids p(t)=(.6-.2t,.3-.1t,.1+.3t), H0 t=(0,.25,.5), H1 t=(.6,.8,1); historical parameterisation unavailable.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='repaired')
    parser.add_argument('--random', type=int, default=400)
    parser.add_argument('--stress', type=int, default=10000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    campaign = Campaign(args.label)
    orders = (math.nextafter(1., 2.), 1 + 1e-12, 1 + 1e-8, 1.01, 2., 10., 1000.)
    score_orders = (1e-8, .1, .5, .9, 1 - 1e-8)
    for i in range(args.random):
        k = 2 + i % 2
        p, q = rng.dirichlet(np.full(k, .6), size=2), rng.dirichlet(np.full(k, .6), size=2)
        campaign.check(categorical(p, q), float(rng.uniform(.001, .9)), f'random-{i}', group='random',
                       order=orders[i % len(orders)], score_order=score_orders[i % 5], vertices=i < 40)
    for i in range(args.stress):
        kind = i % 8
        eps = float(10 ** rng.uniform(-12, -.0001))
        k, n = 2 + i % 2, 1
        if kind == 0:
            p, q = rng.dirichlet(np.full(k, .07), size=1), rng.dirichlet(np.full(k, .07), size=1)
        elif kind == 1:
            rare = float(10 ** rng.uniform(-15, -1))
            p, q = [(rare, 1 - rare)], [(1., 0.)]
        elif kind == 2:
            a = float(rng.uniform(.001, .999))
            p, q = [(a, 1 - a, 0.)], [(0., a, 1 - a)]
        elif kind == 3:
            a = float(10 ** rng.uniform(-16, -5))
            p, q = [(.5, .5)], [(.5 + a, .5 - a)]
        elif kind == 4:
            p = rng.dirichlet(np.full(k, 1.), size=2)
            q = p * (1 - 1e-8) + rng.dirichlet(np.full(k, 1.), size=2) * 1e-8
        elif kind == 5:
            a = float(10 ** rng.uniform(-300, -1))
            p, q = [(a, 1 - a)], [(2 * a, 1 - 2 * a)]
        elif kind == 6:
            p, q, n = [(.999, .001)], [(.001, .999)], 1 + (i // 8) % 10
        else:
            p, q = rng.dirichlet(np.full(k, .1), size=2), rng.dirichlet(np.full(k, .1), size=2)
        campaign.check(categorical(p, q, n), eps, f'stress-{i}', group='stress',
                       order=orders[i % len(orders)], score_order=score_orders[i % 5], score=i % 10 == 0)
        if (i + 1) % 1000 == 0:
            print(json.dumps({'stress_done': i + 1, 'failures': dict(campaign.failures)}), flush=True)
    for k, totals in ((2, range(1, 65)), (3, range(1, 41))):
        p = ((.8, .2), (.7, .3)) if k == 2 else ((.6, .3, .1), (.5, .3, .2))
        q = ((.3, .7), (.2, .8)) if k == 2 else ((.2, .3, .5), (.1, .3, .6))
        curve = []
        for n in totals:
            accepted = campaign.check(categorical(p, q, n), .05, f'count-k{k}-n{n}', group='targeted', score=n <= 10)
            curve.append({'n': n, 'accepted': accepted})
        campaign.refusal_curves[str(k)] = curve
    for k, totals in ((2, (38, 39, 40, 41)), (3, (24, 25, 26, 27))):
        curve = []
        for n in totals:
            uniform = (1. / k,) * k
            accepted = campaign.check(categorical((uniform,), (uniform,), n), .05,
                                      f'uniform-count-k{k}-n{n}', group='targeted', score=False)
            curve.append({'n': n, 'accepted': accepted})
        campaign.refusal_curves[f'uniform-{k}'] = curve
    for n in (1, 2, 3, 5, 8):
        affine = lambda t: (.6 - .2 * t, .3 - .1 * t, .1 + .3 * t)
        campaign.check(categorical(tuple(affine(t) for t in (0, .25, .5)), tuple(affine(t) for t in (.6, .8, 1)), n), .05,
                       f'affine-n{n}', group='targeted')
    for n in (1, 2, 3, 5):
        p = (((.8, .2), (.4, .6)), ((.7, .3), (.4, .6)))
        q = (((.3, .7), (.4, .6)), ((.2, .8), (.4, .6)))
        campaign.check(problem_from_blocks(p, q, (n, n)), .1, f'product-n{n}', group='targeted')
    output = args.output or Path(f'results/composite_testing_validation/{args.label}_summary.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(campaign.summary(), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'summary': str(output), 'counts': dict(campaign.counts), 'failures': dict(campaign.failures)}, indent=2))


if __name__ == '__main__':
    main()
