import datetime
import json
import logging
import os
from typing import Any, Dict


def setup_logger(log_path):
    logger = logging.getLogger("VQE_Engine")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def print_results(results, logger=None):
    output = "\n--- VQE Results ---\n"
    skip_keys = {"final_params", "energy_history"}
    for k, v in results.items():
        if k not in skip_keys:
            output += f"{k:<18}: {v}\n"
    output += "-------------------\n"
    if logger:
        logger.info(output)
    else:
        print(output)


def log_results(exp_dir, exp_name, results, comment="", global_dir=None):
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_name": exp_name,
        "val_energy": results.get("val_energy"),
        "energy_error": results.get("energy_error"),
        "num_params": results.get("num_params"),
        "actual_steps": results.get("actual_steps"),
        "training_seconds": results.get("training_seconds"),
        "comment": comment,
    }
    append_event_jsonl(exp_dir, record)

    # Deprecated in the compact artifact model. Kept in the signature so older
    # call sites do not break while we converge on the new layout.
    _ = global_dir


def summarize_config(config):
    if isinstance(config, dict):
        parts = [f"{k}={v}" for k, v in sorted(config.items())]
        return ", ".join(parts)
    return str(config)


def append_experiment_jsonl(exp_dir: str, record: Dict[str, Any]) -> None:
    path = os.path.join(exp_dir, "events.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        if "schema_version" not in record:
            record["schema_version"] = "1.2"
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def append_event_jsonl(exp_dir: str, record: Dict[str, Any]) -> None:
    append_experiment_jsonl(exp_dir, record)
