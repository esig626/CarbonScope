"""Direct normalisation of the immutable Antoniewicz Table 5 transcription.

This is deliberately a generator, not a second hand-maintained table of atom
positions.  It reads the frozen benchmark's ``TABLE5_REACTIONS`` objects and
converts their labels to normalized atom references.  The library data is
regression-checked against this generator before release.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Mapping

from .schema import AtomRef, AtomTransition


_TABLE5_TO_CANONICAL: Mapping[str, str] = {
    "OAC": "oxaloacetate",
    "AcCoA": "acetyl_coa",
    "citrate": "citrate",
    "AKG": "alpha_ketoglutarate",
    "glutamate": "glutamate",
    "succinate": "succinate",
    "fumarate": "fumarate",
    "aspartate": "aspartate",
    "CO2": "carbon_dioxide",
}


def _frozen_module():
    root = Path(__file__).resolve().parents[3]
    example_dir = root / "examples" / "antoniewicz_tca"
    module_name = "_fluxemu_carbon_transition_frozen_table5"
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.path.insert(0, str(example_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, example_dir / "build_model.py")
        if spec is None or spec.loader is None:  # pragma: no cover - repository invariant
            raise ImportError("could not load frozen Antoniewicz benchmark")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def generated_table5_forward_maps() -> Mapping[str, tuple[AtomTransition, ...]]:
    """Return normalized forward maps generated from frozen label strings."""

    module = _frozen_module()
    generated: dict[str, tuple[AtomTransition, ...]] = {}
    for spec in module.TABLE5_REACTIONS:
        source_labels: dict[str, AtomRef] = {}
        for metabolite_id, labels in spec.substrates:
            canonical = _TABLE5_TO_CANONICAL[metabolite_id]
            source_labels.update(
                {label: AtomRef(canonical, position) for position, label in enumerate(labels, start=1)}
            )
        atom_map: list[AtomTransition] = []
        for metabolite_id, labels in spec.products:
            canonical = _TABLE5_TO_CANONICAL[metabolite_id]
            atom_map.extend(
                AtomTransition(source_labels[label], AtomRef(canonical, position))
                for position, label in enumerate(labels, start=1)
            )
        generated[spec.reaction_id] = tuple(atom_map)
    return generated


__all__ = ["generated_table5_forward_maps"]
