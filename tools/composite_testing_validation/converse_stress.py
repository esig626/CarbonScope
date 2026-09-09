"""Reproducible independent randomized NP audit of both converse APIs.

Run from the repository root with the package installed (or PYTHONPATH=src):
    python tools/composite_testing_validation/converse_stress.py

The reference explicitly enumerates binary multinomial masses with 80-digit
Decimal powers and integer binomial coefficients. It calls neither production
PMFs nor an LP solver. Both accepted public converse results are independently
checked, including cases where the other public API refuses numerically.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random

from fluxemu.exceptions import FluxEMUError
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    CompositeBinaryTestingProblem,
    CompositeMIDLawFamily,
    IndependentMIDProductLaw,
    SimpleBinaryLawPair,
    bruno_converse_at_order,
    composite_renyi_converse_at_order,
)


BASE_SHA = "4f457d2afd75ad1bdffdbe0e9eea25fcaf8a9ad4"
DEFAULT_SEED = 5302026
DEFAULT_CASES = 8000
PRECISION = 80
ABSOLUTE_COMPARISON_TOLERANCE = 2e-12
ORDERS = (math.nextafter(1.0, math.inf), 1.00000001, 1.001, 1.1, 2.0, 10.0, 1e6, 1e300)
TOTALS = (1, 2, 5, 10, 40, 100)
EPSILONS = (math.ulp(0.0), 1e-12, 0.001, 0.1, 0.8, math.nextafter(1.0, 0.0))


def decimal_np_beta(p, q, n, epsilon):
    """Greedy randomized Neyman–Pearson from directly computed Decimal masses."""
    with localcontext() as context:
        context.prec = PRECISION
        p = tuple(Decimal.from_float(value) for value in p)
        q = tuple(Decimal.from_float(value) for value in q)
        masses = []
        for k in range(n + 1):
            coefficient = Decimal(math.comb(n, k))
            pp = coefficient * (p[0]**k if k else 1) * (p[1]**(n-k) if n-k else 1)
            qq = coefficient * (q[0]**k if k else 1) * (q[1]**(n-k) if n-k else 1)
            ratio = qq / pp if pp else Decimal("Infinity")
            masses.append((ratio, pp, qq))
        budget = Decimal.from_float(epsilon)
        beta = Decimal(0)
        for _, pp, qq in sorted(masses, reverse=True):
            reject = min(Decimal(1), budget / pp) if pp else Decimal(1)
            beta += qq * (1 - reject)
            budget -= pp * reject
        return float(beta)


def _inputs(rng, index):
    n = TOTALS[index % len(TOTALS)]
    kind = index % 3
    if kind == 0:
        p = rng.randrange(1, 1024) / 1024
        q = rng.randrange(1, 1024) / 1024
    elif kind == 1:
        p = 10.0 ** (-rng.uniform(1, 323))
        q = 10.0 ** (-rng.uniform(1, 323))
    else:
        p = rng.randrange(1, 1024) / 1024
        q = math.nextafter(p, rng.choice((0.0, 1.0)))
    epsilon = rng.choice(EPSILONS)
    return (p, 1 - p), (q, 1 - q), n, epsilon, ORDERS[index % len(ORDERS)]


def _family(law):
    return CompositeMIDLawFamily(members=(IndependentMIDProductLaw(blocks=(law,)),))


def run(cases=DEFAULT_CASES, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    accepted = Counter()
    refusals = {"composite": Counter(), "simple": Counter()}
    gap_ranges = {name: [math.inf, -math.inf] for name in ("composite", "simple")}
    maximum_excess = {name: None for name in gap_ranges}
    paired_accepted = 0
    for index in range(cases):
        p, q, n, epsilon, order = _inputs(rng, index)
        beta = decimal_np_beta(p, q, n, epsilon)
        null = MultinomialMIDLaw(n, p)
        alternative = MultinomialMIDLaw(n, q)
        problem = CompositeBinaryTestingProblem(null=_family(null), alternative=_family(alternative))
        pair = SimpleBinaryLawPair(null=null, alternative=alternative)
        case_accepted = []
        for name, evaluator, subject in (
            ("composite", composite_renyi_converse_at_order, problem),
            ("simple", bruno_converse_at_order, pair),
        ):
            try:
                result = evaluator(subject, epsilon=epsilon, order=order)
            except FluxEMUError as error:
                refusals[name][type(error).__name__] += 1
                case_accepted.append(False)
                continue
            accepted[name] += 1
            case_accepted.append(True)
            gap = beta - result.type_ii_lower_bound
            previous_minimum = gap_ranges[name][0]
            gap_ranges[name][0] = min(previous_minimum, gap)
            gap_ranges[name][1] = max(gap_ranges[name][1], gap)
            if gap < previous_minimum:
                maximum_excess[name] = {
                    "index": index, "null": p, "alternative": q, "n": n,
                    "epsilon": epsilon, "order": order,
                    "np_beta": beta, "lower_bound": result.type_ii_lower_bound,
                    "excess": -gap,
                }
            if gap < -ABSOLUTE_COMPARISON_TOLERANCE:
                raise AssertionError(f"{name} converse exceeds independent NP: {maximum_excess[name]}")
        paired_accepted += all(case_accepted)
    return {
        "audited_base_sha": BASE_SHA,
        "seed": seed,
        "problems": cases,
        "precision_decimal_digits": PRECISION,
        "oracle": "explicit binary multinomial Decimal masses and randomized Neyman-Pearson",
        "absolute_comparison_tolerance": ABSOLUTE_COMPARISON_TOLERANCE,
        "tolerance_reason": "2e-12 absolute allowance covers accumulated float log/exp roundoff for the largest count 100 and represented simplex residuals; actual maximum excess is reported separately and no values are clipped",
        "orders": ORDERS,
        "count_totals": TOTALS,
        "epsilons": EPSILONS,
        "input_categories": ["dyadic binary probabilities", "positive probabilities down to 1e-323", "adjacent-float nearly identical laws"],
        "accepted_both_apis": paired_accepted,
        "at_least_one_explicit_refusal": cases - paired_accepted,
        "accepted_by_api": dict(accepted),
        "refusals_by_api": {name: dict(counts) for name, counts in refusals.items()},
        "np_beta_minus_bound_gap_ranges": gap_ranges,
        "maximum_excess_cases": maximum_excess,
        "violations": 0,
        "historical_false_converse_reproduced": False,
        "limits": "Decimal PMFs use stored floats without normalization; allowed represented mass residuals can move a near-one NP result a few ulps above one. Singleton binary stress; finite multiclass, structural-zero, ternary and independent-product controls are tested separately. No continuous-family claim or order optimization.",
        "derivation": [
            "For 0<=phi<=1, Holder gives E_Q phi <= epsilon^((lambda-1)/lambda) exp((lambda-1) D_lambda(Q||P)/lambda).",
            "Thus worst-case Type II >= max(0, 1-exp((lambda-1)/lambda*(log(epsilon)+min_{P,Q} D_lambda(Q||P)))).",
            "The simple forward component follows from Holder on 1-phi: beta >= (1-epsilon)^(lambda/(lambda-1)) exp(-D_lambda(P||Q)).",
            "Reverse and forward vertex minimizers need not agree: H0 Bernoulli first-cell probabilities (1/16,1/2), H1 (1/4,3/4), lambda=2 gives reverse pair (1,0) and forward pair (0,0). Composite API intentionally uses only its documented reverse component.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path,
                        default=Path("results/composite_testing_validation/baseline_converse_stress.json"))
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    summary = run(args.cases, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "problems", "accepted_both_apis", "at_least_one_explicit_refusal", "accepted_by_api", "violations",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
