"""Subprocess import firewall checks, kept independent of solver availability."""
import os, subprocess, sys
import pytest
from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis.highs import _highspy

def test_fluxemu_import_does_not_require_highspy():
    code = "import sys; sys.modules['highspy']=None; import fluxemu; import fluxemu.emu.stationary; import fluxemu.emu.transient"
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())

def test_native_module_does_not_load_external_solver_stacks():
    code = "import sys; import fluxemu.flux_analysis.highs; assert not any(x in sys.modules for x in ('cobra','optlang','mfapy','scipy.optimize','highspy'))"
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())

def test_missing_highspy_error_names_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "highspy", None)
    with pytest.raises(AnalysisError, match="optional 'highs' extra"):
        _highspy()
