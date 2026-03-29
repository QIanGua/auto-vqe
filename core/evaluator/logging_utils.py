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
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "timestamp\texp_name\tval_energy\tenergy_error\tnum_params\tactual_steps\ttraining_sec\tcomment\n"
    actual_steps = results.get("actual_steps", "N/A")
    line = (
        f"{ts}\t{exp_name}\t{results['val_energy']:.6f}\t{results['energy_error']:.6f}\t"
        f"{results['num_params']}\t{actual_steps}\t{results['training_seconds']}\t{comment}\n"
    )

    log_path = os.path.join(exp_dir, "results.tsv")
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(header)
        f.write(line)

    if global_dir and global_dir != exp_dir:
        global_path = os.path.join(global_dir, "results.tsv")
        global_exists = os.path.isfile(global_path)
        with open(global_path, "a", encoding="utf-8") as f:
            if not global_exists:
                f.write(header)
            f.write(line)


def summarize_config(config):
    if isinstance(config, dict):
        parts = [f"{k}={v}" for k, v in sorted(config.items())]
        return ", ".join(parts)
    return str(config)


def append_experiment_jsonl(exp_dir: str, record: Dict[str, Any]) -> None:
    path = os.path.join(exp_dir, "results.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        if "schema_version" not in record:
            record["schema_version"] = "1.2"
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
