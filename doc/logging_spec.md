# Auto-VQE 实验日志记录规范

为了实现实验结果的可追溯性、可对比性以及自动化分析，本项目采用统一的结构化日志记录方式。

## 记录文件：`results.jsonl`

每一行代表一次实验运行（一次 VQE 优化过程）。

### 核心字段

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `experiment_id` | `string` | 唯一 UUID |
| `timestamp` | `string` | 运行时间 (`YYYY-MM-DD HH:MM:SS`) |
| `system` | `string` | 物理系统名称 (如 `tfim`, `lih`) |
| `exp_name` | `string` | 实验描述性名称 |
| `n_qubits` | `int` | 量子比特数 |
| `ansatz_spec` | `dict` | **核心**：Ansatz 结构化配置 (见下文) |
| `optimizer_spec` | `dict` | 优化器超参数 ( Adam, lr, max_steps 等) |
| `metrics` | `dict` | 实验结果指标 (能量, 误差, 参数量, 门数量等) |
| `artifact_paths` | `dict` | 关联文件路径 (报告, 图像, JSON 数据) |
| `git_diff` | `string` | 运行时的未提交代码变动 (Patch) |

### AnsatzSpec 规范

所有搜索工具（GA, MultiDim）和 Baseline 必须输出一致的 `ansatz_spec`：

```json
{
  "name": "hea_linear",
  "family": "hea",
  "config": {
    "layers": 2,
    "entanglement": "linear",
    "single_qubit_gates": ["ry"]
  },
  "num_params": 12,
  "metadata": {
    "search": "ga"
  }
}
```

### Metrics 规范

必须包含以下核心指标：
- `val_energy`: 最终能量
- `energy_error`: 与精确值的绝对误差
- `num_params`: 可调参数数量
- `two_qubit_gates`: 二比特门估算数量
- `runtime_sec`: 训练耗时
- `actual_steps`: 实际迭代次数
