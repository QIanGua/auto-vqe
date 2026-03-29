# Agent-VQE 参数映射协议 (Parameter Mapping Protocol)

> **版本**: 1.2  
> **状态**: 当前约定  
> **目标**: 明确在 Ansatz 结构演进（新增层、删除层、门替换、拓扑变化）时，已有参数如何继承、初始化或失效，以确保 Warm-start 的科学性。

---

## 1. 核心原则

1.  **最大化保留 (Maximal Preservation)**: 如果局部结构未变，对应的参数值应当被精确保留。
2.  **局部性 (Locality)**: 结构变化的影响应尽可能局限于变化点，不应导致全线路参数随机化。
3.  **显式标记**: 任何参数映射行为必须在日志中记录映射类型（如 `identity`, `padding`, `interpolation`）。

---

## 2. 场景定义

### 2.1 新增层 (Layer Addition)
- **后置新增 (Appended)**: 前 $L$ 层的参数通过 `Identity` 映射继承，第 $L+1$ 层参数默认以零或小随机值初始化。
- **中间插入 (Inserted)**: 插入点之前的参数继承，插入点之后的参数向后位移。

### 2.2 门替换 (Gate Replacement)
- **同维度替换**: 如 `ry` 替换为 `rx`，参数直接继承。
- **升维替换**: 如 `cnot` (0参数) 替换为 `rzz` (1参数)，新参数初始为 0（退化为等效结构）。

### 2.3 拓扑演进 (Entanglement Evolution)
- 从 `linear` 变为 `full`：
    - 原有的线性对参数保留。
    - 新增的非近邻对比特对参数初始为 0。

---

## 3. 当前实现位置

- 配置级映射：`core/warmstart/config_mapper.py`
- Ansatz 参数映射：`core/warmstart/ansatz_mapper.py`
- 相关测试：`tests/test_parameter_mapper.py`、`tests/test_parameter_mapping.py`

---

## 4. 当前状态与后续增强

- 当前实现已经覆盖配置级和 ansatz 级映射。
- 现有测试覆盖了 identity preservation、层数扩展和基础兼容性。

- 继续扩展 `IdentityMapper` 之外的映射策略。
- 为 `ADAPT-VQE` 提供专门的 `GradientPriorityMapper`。
