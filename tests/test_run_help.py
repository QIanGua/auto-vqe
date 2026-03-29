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


def test_research_runtime_help_runs_as_script():
    result = _run_help("core/research/runtime.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_molecular_generate_help_stays_cli_only():
    result = _run_help("core/molecular/generate.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "openfermion" not in result.stderr.lower()
    assert "pyscf" not in result.stderr.lower()


def test_tfim_ga_search_help_stays_cli_only():
    result = _run_help("experiments/tfim/ga/search.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "starting ga search" not in result.stdout.lower()


def test_tfim_multidim_search_help_stays_cli_only():
    result = _run_help("experiments/tfim/multidim/search.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--- config " not in result.stdout.lower()


def test_lih_ga_search_help_stays_cli_only():
    result = _run_help("experiments/lih/ga/search.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "starting ga search" not in result.stdout.lower()


def test_lih_multidim_search_help_stays_cli_only():
    result = _run_help("experiments/lih/multidim/search.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--- config " not in result.stdout.lower()


def test_tfim_auto_search_help_stays_cli_only():
    result = _run_help("experiments/tfim/orchestration/auto_search.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "starting ga search" not in result.stdout.lower()


def test_lih_auto_search_help_stays_cli_only():
    result = _run_help("experiments/lih/orchestration/auto_search.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "starting ga search" not in result.stdout.lower()
