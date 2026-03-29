import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_help(script_path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_lih_run_help_stays_cli_only():
    result = _run_help("experiments/lih/run.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "tensorcircuit" not in result.stderr.lower()
    assert "qiskit" not in result.stderr.lower()


def test_tfim_run_help_stays_cli_only():
    result = _run_help("experiments/tfim/run.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "tensorcircuit" not in result.stderr.lower()
    assert "qiskit" not in result.stderr.lower()
