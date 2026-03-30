# 分子哈密顿量与数据生成 (Molecular Hamiltonian)

本文档说明如何使用 Agent-VQE 内置的分子哈密顿量生成系统。

## 1. 概览

Agent-VQE 提供了基于 PySCF + OpenFermion 的分子哈密顿量生成工具链：

| 模块 | 路径 | 职责 |
|-----|------|------|
| Builder Spec | `core/molecular/builders.py` | 分子构建器规范 + 几何工厂 |
| Presets | `core/molecular/presets.py` | 内置 H₂ / LiH / BeH₂ 预设 |
| Registry | `core/molecular/registry.py` | 分子构建器注册表 |
| CLI 入口 | `core/molecular/generate.py` | 命令行数据生成 |

## 2. 内置分子体系

| 体系 | 活跃空间 | 默认扫描点 | 说明 |
|-----|---------|-----------|------|
| `h2` | 2 qubit | 0.5 ~ 2.5 Å | 最简单的分子，教学用 |
| `lih` | 4 qubit (STO-3G) | 0.8 ~ 4.0 Å | 含锂键伸缩的基准分子 |
| `beh2` | 6 qubit (STO-3G) | 0.8 ~ 3.0 Å | 含铍的线性分子 |

## 3. CLI 使用

### 列出已注册的分子体系

```bash
uv run python core/molecular/generate.py --list
```

### 使用默认配置生成数据

```bash
uv run python core/molecular/generate.py --system lih
uv run python core/molecular/generate.py --system h2
uv run python core/molecular/generate.py --system beh2
```

输出默认保存到：
```
artifacts/molecular/<system>_pyscf_data.json
```

### 自定义扫描网格

```bash
# 指定几何扫描点（逗号分隔）
uv run python core/molecular/generate.py --system lih --grid 1.0,1.2,1.4,1.6

# 同时指定输出路径
uv run python core/molecular/generate.py --system h2 --grid 0.5,0.74,1.0 --out artifacts/molecular/h2_custom.json
```

### CLI-only Smoke Test

```bash
uv run python core/molecular/generate.py --help
uv run python core/molecular/generate.py --list
```

## 4. 输出数据格式

生成的 `*_pyscf_data.json` 结构如下（以 LiH 为例）：

```json
{
  "system": "lih",
  "basis": "sto-3g",
  "n_qubits": 4,
  "mapping": "jordan_wigner",
  "points": [
    {
      "R": 1.0,
      "nuclear_repulsion": 1.5875,
      "hf_energy": -7.86,
      "active_space_exact_energy": -7.88,
      "full_fci_energy": -7.88,
      "hamiltonian_terms": 100,
      "hamiltonian_pauli_str": "..."
    }
  ]
}
```

### 字段说明

| 字段 | 说明 |
|-----|------|
| `R` | 键长 (Ångström) |
| `nuclear_repulsion` | 核排斥能 |
| `hf_energy` | Hartree-Fock 能量 |
| `active_space_exact_energy` | 活跃空间精确能量（对角化） |
| `full_fci_energy` | 全空间 FCI 能量 |
| `hamiltonian_terms` | 哈密顿量中的 Pauli 项数 |
| `hamiltonian_pauli_str` | 序列化的 Pauli 哈密顿量字符串 |

## 5. LiH 几何扫描

LiH 实验入口提供了专门的几何扫描与可视化命令：

### 执行键长扫描

```bash
uv run python experiments/lih/run.py scan --trials-per-R 1 --max-steps 800 --lr 0.05
```

输出结果：
```text
experiments/lih/artifacts/runs/<timestamp>_lih_geom_scan/lih_geometry_curve_<timestamp>.tsv
```

### 绘制扫描曲线

```bash
# 自动查找最新的 .tsv 文件
uv run python experiments/lih/run.py plot

# 指定 TSV 文件
uv run python experiments/lih/run.py plot path/to/lih_geometry_curve.tsv

# 指定输出目录
uv run python experiments/lih/run.py plot --output-dir path/to/output
```

生成三张图：
1. **能量对比图** — VQE 能量 vs 活跃空间精确能量
2. **Ansatz 误差图** — `|E_VQE - E_active|` vs 键长（对数坐标）
3. **截断误差图** — `|E_active - E_FCI|` vs 键长（对数坐标）

## 6. MindQuantum 对比（可选）

LiH 实验还提供了与 MindQuantum UCCSD 工作流的对比命令：

```bash
uv run python experiments/lih/run.py compare --dist 1.5

# 指定输出 JSON
uv run python experiments/lih/run.py compare --dist 1.5 --save-json artifacts/comparison.json
```

> 注意：需要额外安装 `mindspore` 和 `mindquantum` 依赖。

## 7. 在 Python 中使用

### 使用注册表查询可用体系

```python
from core.molecular.registry import MOLECULAR_REGISTRY

# 列出所有注册体系
for name in MOLECULAR_REGISTRY:
    print(name)
```

### 编程式生成数据

```python
from core.molecular.generate import generate_system_data

# 生成 LiH 数据
data = generate_system_data("lih", grid=[1.0, 1.4, 1.6, 2.0])
```

## 8. 更多参考

详细的分子构建器设计与高级用法见 `doc/molecular_hamiltonian_guide.md`。
