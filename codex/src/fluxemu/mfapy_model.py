"""Shared mfapy model classes used by FluxEMU's in-memory bridge."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ._mfapy import load_mfapy


class BoundaryTargetMetabolicModel(load_mfapy().metabolicmodel.MetabolicModel):
    """Compile isotope propagation into measured terminal products.

    mfapy correctly excludes ``excreted`` metabolites from its steady-state
    balance, but its EMU compiler also drops reactions producing those
    metabolites. That second behavior prevents a terminal observation such as
    Table 5 ``AKG -> glutamate`` from being evaluated without adding an
    artificial export reaction.

    FluxEMU keeps the canonical boundary classification during model/state
    construction. Only while mfapy compiles its generated EMU functions, a
    targeted boundary product is exposed as an isotope intermediate. The
    original role is restored before construction returns. Thus the compiled
    isotope transition contains the real producing reaction while the
    steady-state matrix continues to omit the boundary-product balance row.
    """

    def _targeted_boundary_products(self) -> tuple[str, ...]:
        result: list[str] = []
        for target in self.target_fragments.values():
            if target["use"] != "use":
                continue
            for emu in target["atommap"].replace(" ", "").split("+"):
                matching_ids = [
                    metabolite_id
                    for metabolite_id in self.metabolites
                    if emu.startswith(f"{metabolite_id}_")
                ]
                if not matching_ids:
                    continue
                metabolite_id = max(matching_ids, key=len)
                if (
                    self.metabolites[metabolite_id]["excreted"] == "excreted"
                    and metabolite_id not in result
                ):
                    result.append(metabolite_id)
        return tuple(result)

    @contextmanager
    def _boundary_products_visible_to_emu(self) -> Iterator[None]:
        boundary_products = self._targeted_boundary_products()
        steady_state_metabolite_ids = self.metabolite_ids
        dynamic_metabolite_ids = [
            metabolite_id
            for metabolite_id, metabolite in sorted(
                self.metabolites.items(), key=lambda item: item[1]["order"]
            )
            if metabolite["carbonsource"] != "carbonsource"
            and (
                metabolite["excreted"] != "excreted"
                or metabolite_id in boundary_products
            )
        ]
        self.dynamic_metabolite_ids = tuple(dynamic_metabolite_ids)
        try:
            for metabolite_id in boundary_products:
                self.metabolites[metabolite_id]["excreted"] = "no"
            self.metabolite_ids = dynamic_metabolite_ids
            yield
        finally:
            self.metabolite_ids = steady_state_metabolite_ids
            for metabolite_id in boundary_products:
                self.metabolites[metabolite_id]["excreted"] = "excreted"

    def generate_calmdv(self, mode: str = "normal") -> tuple[str, Any]:
        """Generate stationary and dynamic EMU code with boundary targets."""

        with self._boundary_products_visible_to_emu():
            generated = super().generate_calmdv(mode)

        source_position = {
            source_id: position for position, source_id in enumerate(self.carbon_source)
        }

        def emu_order(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
            definition = item[1]
            atom_positions = tuple(
                int(position) for position in definition["position_list"]
            )
            return (
                source_position[definition["metabolite_name"]],
                len(atom_positions),
                atom_positions,
            )

        self.carbon_source_emu = dict(
            sorted(self.carbon_source_emu.items(), key=emu_order)
        )
        return generated
