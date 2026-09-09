"""Independent finite-testing references; no production enumeration or solver.

PMFs are products of integer multinomial coefficients and Decimal powers.
Tiny problems use exhaustive active-set vertex enumeration, singletons use
randomised Neyman--Pearson, and larger problems use the separately derived dual
LP. Decimal recomputation makes the returned dual objective a feasible bound,
irrespective of the optimiser's status or residual claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import combinations, product
import math

import numpy as np
from scipy.optimize import linprog


PRECISION = 80
D = Decimal


def decimal(value):
    return value if isinstance(value, Decimal) else Decimal.from_float(float(value))


def count_vectors(n, k):
    """Stars-and-bars enumeration independent of production's recursion."""
    for bars in combinations(range(n + k - 1), k - 1):
        boundaries = (-1, *bars, n + k - 1)
        yield tuple(boundaries[i + 1] - boundaries[i] - 1 for i in range(k))


def enumerate_masses(problem):
    """Return full outcomes and unnormalised, unsanitised Decimal PMFs."""
    signature = problem.null.members[0].blocks
    outcomes = tuple(product(*(tuple(count_vectors(b.n, len(b.probabilities))) for b in signature)))
    families = []
    with localcontext() as context:
        context.prec = PRECISION
        for family in (problem.null, problem.alternative):
            rows = []
            for law in family.members:
                row = []
                for outcome in outcomes:
                    mass = D(1)
                    for block, counts in zip(law.blocks, outcome, strict=True):
                        coefficient = math.factorial(block.n)
                        for count in counts:
                            coefficient //= math.factorial(count)
                        block_mass = D(coefficient)
                        for p, count in zip(block.probabilities, counts, strict=True):
                            if count:
                                block_mass *= decimal(p) ** count
                        mass *= block_mass
                    row.append(+mass)
                rows.append(tuple(row))
            families.append(tuple(rows))
    return outcomes, families[0], families[1]


def errors(null, alternative, rejection):
    with localcontext() as context:
        context.prec = PRECISION
        phi = tuple(decimal(x) for x in rejection)
        alpha = tuple(sum((p * x for p, x in zip(row, phi, strict=True)), D(0)) for row in null)
        beta = tuple(sum((q * (1 - x) for q, x in zip(row, phi, strict=True)), D(0)) for row in alternative)
        return alpha, beta


@dataclass(frozen=True)
class OracleResult:
    beta: float
    rejection: tuple[float, ...] | None
    method: str


def neyman_pearson(null, alternative, epsilon):
    """Exact Decimal greedy fractional-knapsack solution for singleton laws."""
    with localcontext() as context:
        context.prec = PRECISION
        p, q = tuple(map(decimal, null)), tuple(map(decimal, alternative))
        ratios = tuple(qi / pi if pi else (D('Infinity') if qi else D(-1)) for pi, qi in zip(p, q, strict=True))
        budget = decimal(epsilon)
        phi = [D(0)] * len(p)
        for i in sorted(range(len(p)), key=ratios.__getitem__, reverse=True):
            if not p[i]:
                phi[i] = D(1) if q[i] else D(0)
            elif budget > 0:
                phi[i] = min(D(1), budget / p[i])
                budget -= p[i] * phi[i]
        _, beta = errors((p,), (q,), phi)
        return OracleResult(float(beta[0]), tuple(map(float, phi)), 'decimal-neyman-pearson')


def _linear_solve(matrix, rhs):
    """Decimal Gaussian elimination with magnitude pivoting."""
    n = len(rhs)
    rows = [list(a) + [b] for a, b in zip(matrix, rhs, strict=True)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(rows[i][j]))
        if not rows[pivot][j]:
            return None
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]
        rows[j] = [x / divisor for x in rows[j]]
        for i in range(n):
            if i != j:
                multiplier = rows[i][j]
                rows[i] = [a - multiplier * b for a, b in zip(rows[i], rows[j], strict=True)]
    return tuple(row[-1] for row in rows)


def vertex_minimax(null, alternative, epsilon):
    """Enumerate every vertex of the finite primal polytope (<=4 outcomes)."""
    with localcontext() as context:
        context.prec = PRECISION
        p = tuple(tuple(map(decimal, row)) for row in null)
        q = tuple(tuple(map(decimal, row)) for row in alternative)
        k = len(p[0])
        if k > 4:
            raise ValueError('vertex oracle is capped at four outcomes')
        eps = decimal(epsilon)
        rows = [tuple(x / eps for x in row) + (D(0),) for row in p]
        rhs = [D(1)] * len(p)
        rows += [tuple(-x for x in row) + (D(-1),) for row in q]
        rhs += [D(-1)] * len(q)
        for i in range(k + 1):
            unit = tuple(D(int(i == j)) for j in range(k + 1))
            rows += [unit, tuple(-x for x in unit)]
            rhs += [D(1), D(0)]
        best = None
        for active in combinations(range(len(rows)), k + 1):
            solution = _linear_solve([rows[i] for i in active], [rhs[i] for i in active])
            if solution is None:
                continue
            if any(sum((a * x for a, x in zip(row, solution, strict=True)), D(0)) > b + D('1e-60') for row, b in zip(rows, rhs, strict=True)):
                continue
            if best is None or solution[-1] < best[-1]:
                best = solution
        if best is None:
            raise ArithmeticError('independent vertex oracle found no feasible vertex')
        return OracleResult(float(best[-1]), tuple(map(float, best[:-1])), 'decimal-vertex-enumeration')


def dual_minimax(null, alternative, epsilon):
    """Solve min eps*sum(u)+sum(v), v >= Q^T*w-P^T*u, sum(w)=1.

    Its value subtracted from one is a minimax Type-II lower bound. The returned
    float is recomputed from genuinely feasible Decimal multipliers: w is
    explicitly rescaled to sum to one, and v is defined as the positive part
    of Q^T*w-P^T*u. This is dual construction, not alteration of any law/test.
    """
    p, q = np.asarray(null, dtype=float), np.asarray(alternative, dtype=float)
    a, k = p.shape
    b = len(q)
    # Let r=epsilon*u so that a tiny epsilon does not vanish as an
    # optimisation objective coefficient. This is an explicit dual change
    # of variables, independent of production's primal solve.
    objective = np.r_[np.ones(a), np.zeros(b), np.ones(k)]
    result = linprog(objective, A_ub=np.c_[-p.T / epsilon, q.T, -np.eye(k)], b_ub=np.zeros(k),
                     A_eq=np.array([[0.] * a + [1.] * b + [0.] * k]), b_eq=[1.],
                     bounds=(0, None), method='highs-ds',
                     options={'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10})
    if not result.success or result.x is None or not np.isfinite(result.x).all() or np.any(result.x < 0):
        raise ArithmeticError('independent dual oracle cannot certify multipliers')
    with localcontext() as context:
        context.prec = PRECISION
        u = tuple(decimal(x) / decimal(epsilon) for x in result.x[:a])
        raw_w = tuple(map(decimal, result.x[a:a + b]))
        total_w = sum(raw_w, D(0))
        if not total_w:
            raise ArithmeticError('independent dual oracle has zero alternative weights')
        w = tuple(x / total_w for x in raw_w)
        slack = [max(D(0), sum((decimal(alternative[j][i]) * w[j] for j in range(b)), D(0)) - sum((decimal(null[j][i]) * u[j] for j in range(a)), D(0))) for i in range(k)]
        lower = D(1) - decimal(epsilon) * sum(u, D(0)) - sum(slack, D(0))
        return OracleResult(float(lower), None, 'independent-dual-lp-feasible-bound')


def solve(null, alternative, epsilon, *, vertices=False):
    if len(null) == len(alternative) == 1:
        return neyman_pearson(null[0], alternative[0], epsilon)
    if vertices:
        return vertex_minimax(null, alternative, epsilon)
    return dual_minimax(null, alternative, epsilon)
