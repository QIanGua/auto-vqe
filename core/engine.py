import torch
import time
import os
import datetime
import logging
import warnings
import contextlib
import io

# Import tensorcircuit once, but silence its optional-backend warnings and
# Python SyntaxWarning noise that may appear on import under newer Python.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import tensorcircuit as tc

# Some tensorcircuit submodules trigger SyntaxWarning under Python 3.12;
# they are harmless for our usage, so we ignore them to keep logs clean.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="tensorcircuit.*")

tc.set_backend("pytorch")


from contextlib import contextmanager

@contextmanager
def experiment_guard(run_py_path, logger=None):
    """
    实验守卫：自动备份 run.py，若实验过程抛出异常则自动回退。
    用于保护 Agent 对 create_circuit 的修改不会因崩溃而丢失原始代码。
    """
    backup = open(run_py_path, encoding="utf-8").read()
    try:
        yield
    except Exception as e:
        with open(run_py_path, "w", encoding="utf-8") as f:
            f.write(backup)
        msg = f"Experiment failed, reverted {run_py_path}. Error: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        raise

def setup_logger(log_path):
    """
    配置标准的日志系统，确保实时落盘且不为空
    """
    logger = logging.getLogger("VQE_Engine")
    logger.setLevel(logging.INFO)
    
    # 清除旧的 handler
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 文件 Handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def vqe_train(
    create_circuit_fn,
    compute_energy_fn,
    n_qubits,
    exact_energy,
    num_params,
    max_steps=1000,
    lr=0.01,
    logger=None,
    seed=None,
    early_stop_window=50,
    early_stop_threshold=1e-8,
    grad_clip_norm=1.0,
):
    """
    通用 VQE 训练循环。

    参数
    ----
    create_circuit_fn : Callable[[Tensor], Tuple[Circuit, int]]
        给定一维参数 Tensor，返回量子线路以及实际使用的参数数量。
    compute_energy_fn : Callable[[Tensor], Tensor]
        输入参数 Tensor，输出能量标量张量。
    n_qubits : int
        量子比特数量（目前仅用于记录，接口保留以便后续扩展）。
    exact_energy : float
        目标基态能量，用于计算 energy_error。
    num_params : int
        需要优化的参数个数。由具体 ansatz 显式给出，避免在此处做脆弱的推断。
    max_steps : int
        优化步数上限。
    lr : float
        Adam 初始学习率。
    logger : logging.Logger | None
        若提供，则使用 logger 记录日志，否则直接 print。
    seed : int | None
        若提供，则在本函数内部设置 torch 随机种子，以提升实验可复现性。
    early_stop_window : int
        早停窗口大小：连续 N 步能量变化小于阈值时自动停止。
    early_stop_threshold : float
        早停阈值：窗口内能量极差低于该值视为收敛。
    grad_clip_norm : float | None
        梯度裁剪范数上限。设为 None 禁用裁剪。
    """
    if seed is not None:
        torch.manual_seed(seed)

    params = torch.randn(num_params, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=100, factor=0.5, min_lr=1e-5
    )

    start_time = time.time()
    energy_history = []
    actual_steps = max_steps

    for i in range(max_steps):
        optimizer.zero_grad()
        energy = compute_energy_fn(params)
        energy.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([params], max_norm=grad_clip_norm)

        optimizer.step()

        e_val = energy.item()
        energy_history.append(e_val)
        scheduler.step(e_val)

        if i % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            msg = f"Step {i}, Energy: {e_val:.6f}, LR: {current_lr:.2e}"
            if logger: logger.info(msg)
            else: print(msg)

        # Early stopping
        if len(energy_history) >= early_stop_window:
            recent = energy_history[-early_stop_window:]
            if max(recent) - min(recent) < early_stop_threshold:
                msg = f"Early stop at step {i}: converged (window={early_stop_window}, threshold={early_stop_threshold:.1e})"
                if logger: logger.info(msg)
                else: print(msg)
                actual_steps = i + 1
                break

    end_time = time.time()
    final_energy = compute_energy_fn(params).item()

    return {
        "val_energy": final_energy,
        "exact_energy": exact_energy,
        "energy_error": abs(final_energy - exact_energy),
        "num_params": num_params,
        "training_seconds": round(end_time - start_time, 2),
        "actual_steps": actual_steps,
        "final_params": params.detach(),
        "energy_history": energy_history,
    }

def print_results(results, logger=None):
    output = "\n--- VQE Results ---\n"
    skip_keys = {"final_params", "energy_history"}
    for k, v in results.items():
        if k not in skip_keys:
            output += f"{k:<18}: {v}\n"
    output += "-------------------\n"
    if logger: logger.info(output)
    else: print(output)

def log_results(exp_dir, exp_name, results, comment=""):
    log_path = os.path.join(exp_dir, "results.tsv")
    header = "timestamp\texp_name\tval_energy\tenergy_error\tnum_params\tactual_steps\ttraining_sec\tcomment\n"
    file_exists = os.path.isfile(log_path)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actual_steps = results.get('actual_steps', 'N/A')
    with open(log_path, "a") as f:
        if not file_exists:
            f.write(header)
        line = f"{ts}\t{exp_name}\t{results['val_energy']:.6f}\t{results['energy_error']:.6f}\t{results['num_params']}\t{actual_steps}\t{results['training_seconds']}\t{comment}\n"
        f.write(line)

def summarize_config(config):
    """
    将 ansatz 配置压缩成一行易读字符串，便于写入日志/tsv。
    """
    if isinstance(config, dict):
        parts = [f"{k}={v}" for k, v in sorted(config.items())]
        return ", ".join(parts)
    return str(config)

def ansatz_search(
    env,
    make_create_circuit_fn,
    config_list,
    exp_dir,
    base_exp_name,
    trials_per_config=3,
    max_steps=1000,
    lr=0.01,
    logger=None,
):
    """
    在给定的 ansatz 配置列表上进行自动搜索。

    - env: QuantumEnvironment 子类实例，提供 n_qubits, exact_energy, compute_energy(c)。
    - make_create_circuit_fn(config) -> (create_circuit_fn, num_params)
    - config_list: 离散的 ansatz 配置列表（dict 或任意可打印对象）。

    Ranking 规则：
      1. 先按 val_energy 从小到大排序；
      2. 若能量差异在 1e-4 以内，则按 num_params 从少到多排序（奥卡姆剃刀）。
    """
    if logger is None:
        # 默认在搜索级别只用一个简单的 logger
        log_path = os.path.join(exp_dir, f"{base_exp_name}_search_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logger = setup_logger(log_path)

    logger.info(f"=== Ansatz Search: {base_exp_name} ===")
    logger.info(f"Target Energy: {env.exact_energy:.6f}")

    best_overall = None
    best_overall_config = None

    for idx, config in enumerate(config_list):
        cfg_str = summarize_config(config)
        logger.info(f"\n--- Config {idx+1}/{len(config_list)}: {cfg_str} ---")

        create_circuit_fn, num_params = make_create_circuit_fn(config)

        def compute_energy_fn(params):
            c, _ = create_circuit_fn(params)
            return env.compute_energy(c)

        best_for_cfg = None

        for t in range(trials_per_config):
            seed = 1000 + idx * 100 + t
            logger.info(f"  Trial {t+1}/{trials_per_config}, seed={seed}")

            results = vqe_train(
                create_circuit_fn=create_circuit_fn,
                compute_energy_fn=compute_energy_fn,
                n_qubits=env.n_qubits,
                exact_energy=env.exact_energy,
                num_params=num_params,
                max_steps=max_steps,
                lr=lr,
                logger=logger,
                seed=seed,
            )

            if (best_for_cfg is None) or (results["val_energy"] < best_for_cfg["val_energy"]):
                best_for_cfg = results

        if best_for_cfg is not None:
            comment = f"config: {cfg_str}"
            log_results(exp_dir, f"{base_exp_name}_cfg{idx}", best_for_cfg, comment=comment)
            logger.info(f"Best for config {idx+1}: val_energy={best_for_cfg['val_energy']:.6f}, num_params={best_for_cfg['num_params']}")
        else:
            logger.warning(f"No successful trials for config {idx+1}")

        def is_better(new, old):
            if old is None:
                return True
            # 先看能量
            if new["val_energy"] + 1e-4 < old["val_energy"]:
                return True
            if abs(new["val_energy"] - old["val_energy"]) <= 1e-4:
                # 能量相近时，参数更少者优先
                return new["num_params"] < old["num_params"]
            return False

        if is_better(best_for_cfg, best_overall):
            best_overall = best_for_cfg
            best_overall_config = config

    logger.info("\n=== Ansatz Search Final Best ===")
    logger.info(f"Best config: {summarize_config(best_overall_config)}")
    print_results(best_overall, logger=logger)

    report_path = generate_report(
        exp_dir,
        f"{base_exp_name}_Best_Report",
        best_overall,
        make_create_circuit_fn(best_overall_config)[0],
        comment=f"Best config: {summarize_config(best_overall_config)}",
    )
    logger.info(f"Report generated at: {report_path}")

    return {
        "best_config": best_overall_config,
        "best_results": best_overall,
        "report_path": report_path,
    }

def generate_report(exp_dir, exp_name, results, create_circuit_fn, comment=""):
    """
    自动生成实验报告：Markdown 分析 + 线路图像可视化
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"report_{timestamp}.md"
    report_path = os.path.join(exp_dir, report_name)
    
    # 1. 尝试生成真实的线路图片 (PNG)
    circuit_img_name = f"circuit_{timestamp}.png"
    circuit_img_path = os.path.join(exp_dir, circuit_img_name)
    has_image = False
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.close('all') # 确保清理之前的绘图对象
        
        c, _ = create_circuit_fn(results['final_params'])
        
        # 1. 优先使用 tensorcircuit 的 draw 方法
        try:
            fig = c.draw(output='mpl')
            # 移除标题以便保持标准的论文风格，或者保留一个简单的标注
            fig.savefig(circuit_img_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            has_image = True
        except Exception as e:
            # 2. 如果标准绘图失败，尝试一个极简的散点分布图作为兜底
            fig, ax = plt.subplots(figsize=(10, 4))
            data = c.to_json()
            if isinstance(data, str):
                import json
                data = json.loads(data)
            
            x = range(len(data))
            y = [d['qubits'][0] for d in data if 'qubits' in d and len(d['qubits']) > 0]
            labels = [d['name'] for d in data]
            
            if y:
                ax.scatter(x[:len(y)], y, marker='s', s=100, color='skyblue')
                for i, txt in enumerate(labels[:30]): 
                    if i < len(y):
                        ax.annotate(txt, (x[i], y[i]), fontsize=8)
            
            ax.set_title(f"Quantum Circuit: {exp_name} (Simple Fallback)")
            ax.set_xlabel("Gate Index")
            ax.set_ylabel("Qubit Index")
            plt.savefig(circuit_img_path, dpi=300)
            plt.close(fig)
            has_image = True
            print(f"Standard draw failed, used fallback. Error: {e}")
    except Exception as e:
        print(f"Image generation failed: {e}")

    # 1.5 绘制收敛曲线
    convergence_img_name = f"convergence_{timestamp}.png"
    convergence_img_path = os.path.join(exp_dir, convergence_img_name)
    has_convergence = False
    energy_history = results.get("energy_history", [])
    if energy_history:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(energy_history, linewidth=0.8, color='#2196F3')
            ax.axhline(y=results['exact_energy'], color='#F44336', linestyle='--', linewidth=1, label=f"Exact: {results['exact_energy']:.6f}")
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Energy")
            ax.set_title(f"{exp_name} Convergence")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(convergence_img_path, dpi=200)
            plt.close(fig)
            has_convergence = True
        except Exception:
            pass

    # 2. 保存完整线路 JSON（仅用于后续分析，不在报告中展示）
    try:
        c, _ = create_circuit_fn(results["final_params"])
        raw_data = c.to_json()
        import json

        if isinstance(raw_data, str):
            circuit_data = json.loads(raw_data)
        else:
            circuit_data = raw_data

        with open(os.path.join(exp_dir, f"circuit_{timestamp}.json"), "w") as f:
            f.write(json.dumps(circuit_data, indent=2))
    except Exception:
        # JSON 导出失败不会影响报告生成
        pass

    # 3. 生成报告文本
    accuracy_status = "探索中"
    if results['energy_error'] < 0.0016:
        accuracy_status = "成功 (达到化学精度)"
    if results['energy_error'] < 1e-5:
        accuracy_status = "完美收敛 (高精度)"

    img_embed = f"![Circuit Diagram]({circuit_img_name})\n" if has_image else "*线路图生成暂不可用，请查看 JSON 结构数据。*\n"

    convergence_embed = f"![Convergence Curve]({convergence_img_name})\n" if has_convergence else ""
    actual_steps = results.get('actual_steps', 'N/A')

    report_content = f"""# 实验结题报告: {exp_name}
- **日期**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **后端**: TensorCircuit (PyTorch)

## 一、 核心指标
| 指标 | 数值 |
| :--- | :--- |
| **最终能量** | {results['val_energy']:.8f} |
| **精确能量** | {results['exact_energy']:.8f} |
| **能量误差** | {results['energy_error']:.8e} |
| **参数量** | {results['num_params']} |
| **实际步数** | {actual_steps} |
| **耗时** | {results['training_seconds']} s |
| **状态** | {accuracy_status} |

## 二、 收敛曲线
{convergence_embed}
## 三、 线路可视化图示
{img_embed}

## 四、 结果分析
{comment if comment else "自动生成的分析报告：实验已完成，收敛曲线正常。"}

### 精度评价
{'当前结果已进入化学精度范围。' if results['energy_error'] < 0.0016 else '当前结果尚未达到化学精度，建议压榨参数或增加深度。'}

---
*完整实验数据（包括线路 JSON 与图像）已保存至目录下。*
"""
    with open(report_path, "w") as f:
        f.write(report_content)
    
    return report_path
