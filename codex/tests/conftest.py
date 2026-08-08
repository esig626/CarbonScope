"""Shared test-suite resource hygiene for the constrained workspace runner."""

from __future__ import annotations

import gc

import pytest


@pytest.fixture(autouse=True)
def collect_closed_figure_cycles(monkeypatch):
    """Bound test-render memory and release closed figures between tests."""

    from matplotlib.figure import Figure

    original_savefig = Figure.savefig

    def memory_bounded_savefig(figure, *args, **kwargs):
        dpi = kwargs.get("dpi")
        if isinstance(dpi, (int, float)) and dpi > 100:
            kwargs["dpi"] = 100
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", memory_bounded_savefig)

    yield
    gc.collect()
