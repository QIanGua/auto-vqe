# Molecular Hamiltonian Guide

> 更新时间：2026-03-29
> 适用范围：`core/molecular/` 下的通用分子哈密顿量生成层

## 1. 目标

当前项目已经不再把分子体系哈密顿量生成逻辑只放在 `experiments/lih/` 里。

新的通用入口位于 `core/molecular/`，它负责：

- 描述一个分子体系的 geometry scan
- 描述 active space 和 qubit mapping
- 统一调用 PySCF + OpenFermion 生成 qubit Hamiltonian
- 统一保存成 `MolecularHamiltonianData`
- 通过 registry 暴露成可复用 builder

这意味着：

- `LiH` 只是一个 preset
- `H2`、`BeH2` 可以走同一套生成链路
- 新分子体系不需要从头复制一份 `pyscf_generate_xxx.py`

## 2. 当前模块结构

核心文件：

- `core/molecular/schema.py`
  - `PauliTerm`
  - `MolecularHamiltonianPoint`
  - `MolecularHamiltonianData`
- `core/molecular/pauli.py`
  - 从 Pauli 列表恢复精确本征值
- `core/molecular/loader.py`
  - 加载 JSON
  - 选最近坐标点
  - 选最低 exact-energy 点
- `core/molecular/serialize.py`
  - 保存统一数据格式
- `core/molecular/builders.py`
  - `MolecularBuilderSpec`
  - geometry factory
  - scan dataset builder
- `core/molecular/presets.py`
  - 内置 `H2` / `LiH` / `BeH2`
- `core/molecular/registry.py`
  - 注册和获取 builder
- `core/molecular/generate.py`
  - CLI 入口

## 3. 当前调用链

统一生成链路：

```text
registry/preset
  -> MolecularBuilderSpec
  -> build_scan_dataset(...)
  -> PySCF / OpenFermion
  -> MolecularHamiltonianData
  -> save_molecular_hamiltonian_data(...)
```

对于 LiH，当前脚本：

- `experiments/lih/data/pyscf_generate_lih_data.py`

已经只是一个薄封装：

- 从 registry 取 `lih`
- 用实验侧 grid 覆盖坐标列表
- 调 `generate_dataset(...)`
- 落盘到 `experiments/lih/data/lih_pyscf_data.json`

## 4. 如何使用 CLI

列出当前已注册体系：

```bash
uv run python core/molecular/generate.py --list
```

生成默认 preset 数据：

```bash
uv run python core/molecular/generate.py --system lih
uv run python core/molecular/generate.py --system h2
uv run python core/molecular/generate.py --system beh2
```

覆盖坐标扫描网格：

```bash
uv run python core/molecular/generate.py --system lih --grid 1.0,1.2,1.4,1.6
uv run python core/molecular/generate.py --system h2 --grid 0.5,0.74,1.0
```

指定输出位置：

```bash
uv run python core/molecular/generate.py --system beh2 --out artifacts/molecular/beh2_scan.json
```

默认输出路径：

```text
artifacts/molecular/<system>_pyscf_data.json
```

## 5. 当前内置 preset

当前 registry 内置：

- `h2`
- `lih`
- `beh2`

定义位置：

- `core/molecular/presets.py`

说明：

- `h2`
  - 线性双原子
  - `sto-3g`
  - 最小 active-space preset
- `lih`
  - 线性双原子
  - `sto-3g`
  - 当前与仓库 LiH 4-qubit 实验兼容
- `beh2`
  - 对称线性三原子
  - `sto-3g`
  - 用于说明 builder 不只支持双原子

## 6. 如何从代码里使用

### 6.1 直接按名字生成

```python
from core.molecular.generate import generate_dataset

data = generate_dataset("lih", coordinate_values=[1.0, 1.2, 1.4, 1.6])
```

### 6.2 先取 builder 再局部覆盖 grid

```python
from core.molecular.builders import build_scan_dataset, clone_builder_with_coordinates
from core.molecular.registry import get_molecular_builder

builder = get_molecular_builder("h2")
builder = clone_builder_with_coordinates(builder, [0.5, 0.74, 1.0])
data = build_scan_dataset(builder)
```

### 6.3 读取 builder 元信息

```python
from core.molecular.builders import builder_summary
from core.molecular.registry import get_molecular_builder

summary = builder_summary(get_molecular_builder("beh2"))
```

## 7. 如何新增一个分子体系

推荐流程：

1. 在 `core/molecular/presets.py` 新增一个 builder 构造函数
2. 如果是简单线性体系，优先复用：
   - `make_diatomic_geometry_factory(...)`
   - `make_symmetric_triatomic_geometry_factory(...)`
3. 指定：
   - `MolecularProblemSpec`
   - `MolecularActiveSpaceSpec`
   - `coordinate_axis`
   - `coordinate_values`
   - `geometry_factory`
4. 在 `core/molecular/registry.py` 注册名称
5. 用 `core/molecular/generate.py --system <name>` 验证

最小示例：

```python
from core.molecular.builders import MolecularBuilderSpec, make_diatomic_geometry_factory
from core.molecular.generator import MolecularActiveSpaceSpec, MolecularProblemSpec

spec = MolecularBuilderSpec(
    problem=MolecularProblemSpec(
        system="NaH",
        geometry=[("Na", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        description="NaH scan",
    ),
    active_space=MolecularActiveSpaceSpec(
        occupied_indices=[0],
        active_indices=[1, 2],
        mapping="jordan_wigner",
    ),
    coordinate_axis="R",
    coordinate_unit="Angstrom",
    coordinate_values=[1.0, 1.2, 1.5, 1.8],
    geometry_factory=make_diatomic_geometry_factory("Na", "H"),
)
```

然后注册：

```python
from core.molecular.registry import register_molecular_builder

register_molecular_builder("nah", spec)
```

## 8. 兼容性说明

当前 `LiHEnvironment` 仍保留两层兼容：

- 可以读取新的 `MolecularHamiltonianData` 风格 JSON
- 也仍支持旧测试里直接注入 legacy dict cache

因此当前改造不会破坏：

- `tests/test_lih_env.py`
- 旧的 LiH 数据文件读取路径
- 数据缺失时的 toy fallback

## 9. 当前边界

当前 builder 层仍然偏向：

- 1 维 geometry scan
- 预定义 active-space
- `PySCF + OpenFermion + Jordan-Wigner`

也就是说它已经足够解决：

- `H2`
- `LiH`
- `BeH2`
- 同类的小规模扫描体系

但还没有进一步抽象成：

- 多维几何参数扫描
- 多种 qubit mapping backend 插件化
- 与实验搜索配置自动联动的数据缓存管理

这些可以作为下一步演进方向。
