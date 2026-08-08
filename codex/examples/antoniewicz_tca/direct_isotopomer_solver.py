"""Independent isotopomer balances for Antoniewicz et al. Table 5.

This module intentionally imports neither FluxEMU nor mfapy.  It enumerates
the full isotopomer state for the six balanced/product pools and solves the
176-variable stationary balance directly with NumPy.
"""

from __future__ import annotations

from itertools import product
from typing import Mapping

import numpy as np


FLUXES: Mapping[str, float] = {
    "v1": 100.0,
    "v2": 100.0,
    "v3": 50.0,
    "v4": 50.0,
    "v5": 50.0,
    "v6": 125.0,
    "v7": 75.0,
    "v8": 50.0,
}


def isotopomers(carbon_count: int) -> tuple[tuple[int, ...], ...]:
    """Return atom-ordered binary carbon isotopomers."""

    return tuple(product((0, 1), repeat=carbon_count))


def source_distribution(carbon_count: int, fractions: Mapping[str, float]) -> np.ndarray:
    """Convert #010-style source fractions into an ordered isotopomer vector."""

    states = isotopomers(carbon_count)
    result = np.zeros(len(states), dtype=float)
    for pattern, fraction in fractions.items():
        if not pattern.startswith("#") or len(pattern) != carbon_count + 1:
            raise ValueError(f"invalid {carbon_count}-carbon isotopomer {pattern!r}")
        result[states.index(tuple(int(bit) for bit in pattern[1:]))] = float(fraction)
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("source isotopomer fractions must sum to one")
    return result


ACCOA = source_distribution(2, {"#00": 0.50, "#01": 0.25, "#11": 0.25})
ASPARTATE = source_distribution(4, {"#0000": 1.0})


def transition_matrix(
    source_carbons: int,
    destination_carbons: int,
    positions: tuple[int, ...],
) -> np.ndarray:
    """Return the atom-transition matrix for one deterministic transition."""

    source = isotopomers(source_carbons)
    destination = isotopomers(destination_carbons)
    matrix = np.zeros((len(destination), len(source)), dtype=float)
    for column, state in enumerate(source):
        output = tuple(state[position] for position in positions)
        matrix[destination.index(output), column] = 1.0
    return matrix


def reverse_orientation(carbon_count: int) -> np.ndarray:
    """Return the atom-order reversal used by the paper's symmetric molecules."""

    return transition_matrix(carbon_count, carbon_count, tuple(reversed(range(carbon_count))))


def symmetric_orientation(carbon_count: int) -> np.ndarray:
    """Implement the exact 0.5 abcd + 0.5 dcba branches in Table 5."""

    size = 2**carbon_count
    return 0.5 * (np.eye(size) + reverse_orientation(carbon_count))


def condensation_with_accoa() -> np.ndarray:
    """Return v1: OAC(abcd) + AcCoA(ef) -> citrate(dcbfea)."""

    oac_states = isotopomers(4)
    accoa_states = isotopomers(2)
    citrate_states = isotopomers(6)
    matrix = np.zeros((len(citrate_states), len(oac_states)), dtype=float)
    for column, oac in enumerate(oac_states):
        for accoa_index, accoa in enumerate(accoa_states):
            citrate = (oac[3], oac[2], oac[1], accoa[1], accoa[0], oac[0])
            matrix[citrate_states.index(citrate), column] += ACCOA[accoa_index]
    return matrix


def solve_stationary(*, symmetric: bool = True) -> dict[str, np.ndarray]:
    """Solve the published stationary isotope balances numerically.

    ``symmetric=False`` removes Table 5's reverse orientation only and is used
    by the negative regression test.
    """

    flux = FLUXES
    oac_size, citrate_size, akg_size, glutamate_size, c4_size = 16, 64, 32, 32, 16
    starts = {
        "OAC": 0,
        "citrate": oac_size,
        "AKG": oac_size + citrate_size,
        "glutamate": oac_size + citrate_size + akg_size,
        "succinate": oac_size + citrate_size + akg_size + glutamate_size,
        "fumarate": oac_size + citrate_size + akg_size + glutamate_size + c4_size,
    }
    dimension = sum((oac_size, citrate_size, akg_size, glutamate_size, c4_size, c4_size))
    system = np.zeros((dimension, dimension), dtype=float)
    rhs = np.zeros(dimension, dtype=float)

    eye4, eye6, eye5 = np.eye(16), np.eye(64), np.eye(32)
    symmetry = symmetric_orientation(4) if symmetric else eye4
    condense = condensation_with_accoa()
    citrate_to_akg = transition_matrix(6, 5, (0, 1, 2, 3, 4))
    akg_to_succinate = transition_matrix(5, 4, (1, 2, 3, 4))

    def block(name: str) -> slice:
        start = starts[name]
        length = {"OAC": 16, "citrate": 64, "AKG": 32, "glutamate": 32, "succinate": 16, "fumarate": 16}[name]
        return slice(start, start + length)

    # (v1 + v7) OAC - v6 * symmetry(Fum) = v8 * Asp
    system[block("OAC"), block("OAC")] = (flux["v1"] + flux["v7"]) * eye4
    system[block("OAC"), block("fumarate")] = -flux["v6"] * symmetry
    rhs[block("OAC")] = flux["v8"] * ASPARTATE

    # v2 Cit - v1 * condensation(OAC, AcCoA) = 0
    system[block("citrate"), block("citrate")] = flux["v2"] * eye6
    system[block("citrate"), block("OAC")] = -flux["v1"] * condense

    # (v3 + v4) AKG - v2 * Cit[12345] = 0
    system[block("AKG"), block("AKG")] = (flux["v3"] + flux["v4"]) * eye5
    system[block("AKG"), block("citrate")] = -flux["v2"] * citrate_to_akg

    # v3 Glu - v3 AKG = 0
    system[block("glutamate"), block("glutamate")] = flux["v3"] * eye5
    system[block("glutamate"), block("AKG")] = -flux["v3"] * eye5

    # v5 Suc - v4 AKG[2345] = 0
    system[block("succinate"), block("succinate")] = flux["v5"] * eye4
    system[block("succinate"), block("AKG")] = -flux["v4"] * akg_to_succinate

    # v6 Fum - v5 * symmetry(Suc) - v7 * symmetry(OAC) = 0
    system[block("fumarate"), block("fumarate")] = flux["v6"] * eye4
    system[block("fumarate"), block("succinate")] = -flux["v5"] * symmetry
    system[block("fumarate"), block("OAC")] = -flux["v7"] * symmetry

    solution = np.linalg.solve(system, rhs)
    return {name: solution[block(name)] for name in starts}


def mass_isotopomer_distribution(isotopomer_distribution: np.ndarray, carbon_count: int) -> np.ndarray:
    """Collapse an atom-ordered isotopomer vector into its complete MID."""

    result = np.zeros(carbon_count + 1, dtype=float)
    for fraction, state in zip(isotopomer_distribution, isotopomers(carbon_count)):
        result[sum(state)] += fraction
    return result


def glutamate_mid(*, symmetric: bool = True) -> np.ndarray:
    """Return the complete direct-solver glutamate MID."""

    return mass_isotopomer_distribution(solve_stationary(symmetric=symmetric)["glutamate"], 5)


__all__ = [
    "ACCOA",
    "ASPARTATE",
    "FLUXES",
    "glutamate_mid",
    "mass_isotopomer_distribution",
    "solve_stationary",
]
