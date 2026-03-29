import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_help(script_path: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, script_path, *args, "--help"],
        cwd=REPO_ROOT,
        env=env,
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


def test_render_report_help_runs_as_script():
    result = _run_help("core/evaluator/render_report.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_tfim_search_help_stays_cli_only():
    result = _run_help("experiments/tfim/run.py", "search")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "ga" in result.stdout.lower()
    assert "multidim" in result.stdout.lower()


def test_tfim_baseline_help_stays_cli_only():
    result = _run_help("experiments/tfim/run.py", "baseline")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "tensorcircuit" not in result.stderr.lower()


def test_lih_search_help_stays_cli_only():
    result = _run_help("experiments/lih/run.py", "search")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "ga" in result.stdout.lower()
    assert "multidim" in result.stdout.lower()


def test_lih_baseline_help_stays_cli_only():
    result = _run_help("experiments/lih/run.py", "baseline")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "tensorcircuit" not in result.stderr.lower()


def test_tfim_auto_search_help_stays_cli_only():
    result = _run_help("experiments/tfim/run.py", "auto")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "starting ga search" not in result.stdout.lower()


def test_tfim_scale_100q_help_stays_cli_only():
    result = _run_help("experiments/tfim/run.py", "scale-100q")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "scipy" not in result.stderr.lower()


def test_lih_auto_search_help_stays_cli_only():
    result = _run_help("experiments/lih/run.py", "auto")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "starting ga search" not in result.stdout.lower()


def test_lih_plot_help_stays_cli_only():
    result = _run_help("experiments/lih/run.py", "plot")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "matplotlib" not in result.stderr.lower()


def test_lih_compare_help_stays_cli_only():
    result = _run_help("experiments/lih/run.py", "compare")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "mindquantum" not in result.stderr.lower()
    assert "mindspore" not in result.stderr.lower()
