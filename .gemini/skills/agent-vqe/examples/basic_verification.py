"""
Agent-VQE 基础验证示例

展示如何以编程方式运行 VQE 验证实验。
"""
import os
import sys
import json

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def run_tfim_verification(trials: int = 2):
    """运行 TFIM 基础验证."""
    from experiments.tfim.env import ENV
    from experiments.shared import load_best_config, run_config_experiment
    from experiments.tfim.run import MANIFEST

    print("=" * 60)
    print("TFIM 基础验证")
    print(f"  量子比特数: {ENV.n_qubits}")
    print(f"  精确能量:   {ENV.exact_energy:.8f}")
    print("=" * 60)

    # 获取最优配置
    config, config_path = load_best_config(MANIFEST)
    print(f"  使用配置: {config_path or '(fallback)'}")
    print(f"  配置内容: {json.dumps(config, indent=2)}")

    # 运行实验
    run_config_experiment(MANIFEST, trials=trials)


def run_lih_verification(trials: int = 2):
    """运行 LiH 基础验证."""
    from experiments.lih.env import ENV
    from experiments.shared import load_best_config, run_config_experiment
    from experiments.lih.run import MANIFEST

    print("=" * 60)
    print("LiH 基础验证")
    print(f"  量子比特数: {ENV.n_qubits}")
    print(f"  精确能量:   {ENV.exact_energy:.8f}")
    print("=" * 60)

    config, config_path = load_best_config(MANIFEST)
    print(f"  使用配置: {config_path or '(fallback)'}")
    print(f"  配置内容: {json.dumps(config, indent=2)}")

    run_config_experiment(MANIFEST, trials=trials)


def inspect_run_json(run_dir: str):
    """检查一个运行目录中的 run.json."""
    run_json_path = os.path.join(run_dir, "run.json")
    if not os.path.exists(run_json_path):
        print(f"未找到 run.json: {run_json_path}")
        return

    with open(run_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print(f"运行报告: {run_dir}")
    print(f"  Schema 版本:  {data.get('schema_version')}")
    print(f"  实验名:      {data.get('exp_name')}")
    print(f"  时间戳:      {data.get('timestamp')}")

    metrics = data.get("metrics", {})
    print(f"  能量值:      {metrics.get('val_energy', 'N/A')}")
    print(f"  精确能量:    {metrics.get('exact_energy', 'N/A')}")
    print(f"  能量误差:    {metrics.get('energy_error', 'N/A')}")
    print(f"  参数量:      {metrics.get('num_params', 'N/A')}")
    print(f"  双比特门:    {metrics.get('two_qubit_gates', 'N/A')}")
    print(f"  运行时间:    {metrics.get('runtime_sec', 'N/A')} sec")

    git = data.get("git_info", {})
    print(f"  Git Commit:  {git.get('commit', 'N/A')}")
    print(f"  Git Dirty:   {git.get('dirty', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent-VQE 基础验证示例")
    parser.add_argument("--system", choices=["tfim", "lih", "both"], default="both")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--inspect", type=str, help="检查指定运行目录的 run.json")
    args = parser.parse_args()

    if args.inspect:
        inspect_run_json(args.inspect)
    elif args.system in ("tfim", "both"):
        run_tfim_verification(trials=args.trials)
    if args.system in ("lih", "both") and not args.inspect:
        run_lih_verification(trials=args.trials)
