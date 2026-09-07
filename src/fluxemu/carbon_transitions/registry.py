"""Load FluxEMU's authoritative curated carbon-transition library."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import yaml

from fluxemu.exceptions import MappingError

from .schema import CarbonTransition, MetaboliteDefinition
from .validator import validate_library


DATA_FILENAMES = (
    "glycolysis.yaml",
    "pyruvate_lactate.yaml",
    "pentose_phosphate.yaml",
    "tca.yaml",
    "anaplerosis.yaml",
    "glutaminolysis.yaml",
    "malate_aspartate.yaml",
    "citrate_acetylcoa.yaml",
    "acetate.yaml",
    "serine_glycine.yaml",
)


def _yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise MappingError(f"could not parse transition data {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise MappingError(f"transition data {path} must be a YAML mapping")
    return value


class TransitionLibrary:
    """Validated immutable reaction and metabolite-carbon registry."""

    def __init__(
        self,
        metabolites: Iterable[MetaboliteDefinition],
        transitions: Iterable[CarbonTransition],
    ) -> None:
        metabolite_items = tuple(metabolites)
        metabolite_ids = [item.canonical_id for item in metabolite_items]
        if len(metabolite_ids) != len(set(metabolite_ids)):
            raise MappingError("metabolite carbon registry has duplicate canonical_id values")
        self.metabolites = MappingProxyType({item.canonical_id: item for item in metabolite_items})
        self.transitions = tuple(transitions)
        validate_library(self.transitions, self.metabolites)
        transition_ids = [item.canonical_id for item in self.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            raise MappingError("carbon-transition library has duplicate canonical_id values")
        self.by_id = MappingProxyType({item.canonical_id: item for item in self.transitions})

    @classmethod
    def from_directory(cls, directory: str | Path) -> "TransitionLibrary":
        root = Path(directory)
        metabolites_document = _yaml_mapping(root / "metabolites.yaml")
        if metabolites_document.get("schema_version") != 1:
            raise MappingError("metabolites.yaml must declare schema_version: 1")
        raw_metabolites = metabolites_document.get("metabolites")
        if not isinstance(raw_metabolites, list):
            raise MappingError("metabolites.yaml must contain a metabolites list")
        metabolites = tuple(
            MetaboliteDefinition.from_dict(item, f"metabolites[{index}]")
            for index, item in enumerate(raw_metabolites)
        )
        transitions: list[CarbonTransition] = []
        for filename in DATA_FILENAMES:
            path = root / filename
            if not path.is_file():
                raise MappingError(f"transition data file is missing: {path}")
            document = _yaml_mapping(path)
            if document.get("schema_version") != 1:
                raise MappingError(f"{filename} must declare schema_version: 1")
            raw_entries = document.get("entries")
            if not isinstance(raw_entries, list):
                raise MappingError(f"{filename} must contain an entries list")
            transitions.extend(
                CarbonTransition.from_dict(item, f"{filename}.entries[{index}]")
                for index, item in enumerate(raw_entries)
            )
        return cls(metabolites, transitions)


@lru_cache(maxsize=1)
def load_default_library() -> TransitionLibrary:
    """Load and validate the packaged Version-1 transition data once."""

    return TransitionLibrary.from_directory(Path(files(__package__) / "data"))


__all__ = ["TransitionLibrary", "load_default_library"]
