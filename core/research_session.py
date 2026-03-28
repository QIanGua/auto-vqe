import os
import json
import datetime
from typing import Any, Dict, List, Optional

class ResearchSession:
    """
    Manages the autoresearch.jsonl and autoresearch.md files for an autonomous agent.
    """
    def __init__(self, system_dir: str, state_dir: Optional[str] = None):
        self.system_dir = system_dir
        self.state_dir = state_dir or system_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.state_dir, "autoresearch.jsonl")
        self.md_path = os.path.join(self.state_dir, "autoresearch.md")
        
    def log_decision(self, 
                     iteration: int, 
                     hypothesis: str, 
                     action: str, 
                     results: Dict[str, Any], 
                     decision: str, 
                     rationale: str):
        """
        Appends a decision record to autoresearch.jsonl.
        """
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "iteration": iteration,
            "hypothesis": hypothesis,
            "action": action,
            "results": results,
            "decision": decision,
            "rationale": rationale,
            "schema_version": "1.0"
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    def update_brain(self, 
                     objective: str, 
                     best_config: Dict[str, Any], 
                     best_energy_error: float,
                     dead_ends: List[str], 
                     next_hypotheses: List[str]):
        """
        Updates the autoresearch.md 'brain' file.
        """
        content = f"""# Research Brain: {os.path.basename(self.system_dir)}

## Objective
{objective}

## Best Known Configuration
- **Energy Error**: {best_energy_error:.2e}
- **Config**: `{json.dumps(best_config)}`

## Dead Ends
{chr(10).join([f"- {de}" for de in dead_ends])}

## Next Hypotheses
{chr(10).join([f"- {nh}" for nh in next_hypotheses])}

## Iteration History
(See autoresearch.jsonl for full details)
"""
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(content)

    def get_latest_iteration(self) -> int:
        if not os.path.exists(self.jsonl_path):
            return 0
        iterations = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    iterations.append(data.get("iteration", 0))
                except:
                    continue
        return max(iterations) if iterations else 0

    def get_best_performance(self) -> float:
        best_err = float('inf')
        if not os.path.exists(self.jsonl_path):
            return best_err
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    err = data.get("results", {}).get("energy_error", float('inf'))
                    if err < best_err:
                        best_err = err
                except:
                    continue
        return best_err
