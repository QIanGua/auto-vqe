import os
import sys
import json
import shutil
import tempfile
import torch
import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import vqe_train, generate_report
from core.circuit_factory import build_ansatz

def test_audit_logging():
    print("Starting Audit Logging Smoke Test...")
    
    # 1. Setup temporary experiment directory
    test_dir = tempfile.mkdtemp(prefix="vqe_smoke_test_")
    print(f"Using temp dir: {test_dir}")
    
    try:
        # 2. Minimal Ansatz & Environment setup
        n_qubits = 2
        config = {
            "layers": 1,
            "single_qubit_gates": ["ry"],
            "two_qubit_gate": "cnot",
            "entanglement": "linear"
        }
        config_path = os.path.join(test_dir, "mock_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
            
        create_circuit_fn, num_params = build_ansatz(config, n_qubits)
        
        def compute_energy_fn(params):
            # Mock energy: sum of squares (minimum at 0)
            return torch.sum(params**2)
            
        # 3. Run a tiny training session
        print("Running tiny VQE training...")
        results = vqe_train(
            create_circuit_fn=create_circuit_fn,
            compute_energy_fn=compute_energy_fn,
            n_qubits=n_qubits,
            exact_energy=0.0,
            num_params=num_params,
            max_steps=5, # Tiny!
            lr=0.1,
            seed=42
        )
        
        # 4. Generate report (this triggers the JSONL append with audit info)
        print("Generating report and logging audit info...")
        report_path = generate_report(
            exp_dir=test_dir,
            exp_name="SmokeTestExp",
            results=results,
            create_circuit_fn=create_circuit_fn,
            ansatz_spec=config,
            config_path=config_path
        )
        
        # 5. Verify results.jsonl contents
        jsonl_path = os.path.join(test_dir, "results.jsonl")
        if not os.path.exists(jsonl_path):
            raise RuntimeError("results.jsonl was not created!")
            
        with open(jsonl_path, "r") as f:
            record = json.loads(f.readline())
            
        print("\nVerifying JSONL Record Fields:")
        expected_fields = [
            "schema_version", 
            "git_info", 
            "runtime_env", 
            "config_path_used",
            "metrics",
            "artifact_paths"
        ]
        
        for field in expected_fields:
            if field in record:
                print(f"  [PASS] Field '{field}' found.")
            else:
                print(f"  [FAIL] Field '{field}' MISSING!")
                
        # Specific sub-field checks
        if record.get("schema_version") == "1.2":
            print("  [PASS] Schema version is 1.2")
        else:
            print(f"  [FAIL] schema_version is {record.get('schema_version')} (expected 1.2)")
            
        git_info = record.get("git_info", {})
        if "commit" in git_info:
            print("  [PASS] Git commit capture works.")
            
        if "diff_path" in git_info:
            print(f"  [PASS] Git diff optimized to file: {git_info['diff_path']}")
            full_patch_path = os.path.join(test_dir, git_info['diff_path'])
            if os.path.exists(full_patch_path):
                print(f"  [PASS] Patch file exists at {full_patch_path}")
            else:
                print(f"  [FAIL] Patch file MISSING at {full_patch_path}")
        
        if "diff" in git_info and git_info["diff"]:
            print("  [FAIL] Full 'diff' text STILL in JSONL!")
        else:
            print("  [PASS] Full 'diff' text removed from JSONL.")
            
        if "python_version" in record.get("runtime_env", {}):
            print("  [PASS] Runtime env capture works.")
            
        if record.get("config_path_used") == config_path:
            print("  [PASS] Config path tracking works.")
            
        # 6. Verify Report.md content
        with open(report_path, "r") as f:
            report_text = f.read()
            if "## 五、 审计信息" in report_text:
                print("  [PASS] Report contains Audit Info section.")
            else:
                print("  [FAIL] Report MISSING Audit Info section.")

        print("\nSmoke Test Completed Successfully!")
        
    finally:
        shutil.rmtree(test_dir)
        print(f"Cleaned up temp dir: {test_dir}")

if __name__ == "__main__":
    test_audit_logging()
