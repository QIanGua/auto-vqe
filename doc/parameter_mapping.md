# Agent-VQE 参数映射协议 (Parameter Mapping Protocol)

> **版本**: 1.0 (P1)  
> **状态**: 草案  
> **目标**: 明确在 Ansatz 结构演进（新增层、删除层、门替换、拓扑变化）时，已有参数如何继承、初始化或失效，以确保 Warm-start 的科学性。

---

## 1. 核心原则

1.  **最大化保留 (Maximal Preservation)**: 如果局部结构未变，对应的参数值应当被精确保留。
2.  **局部局部性 (Locality)**: 结构变化的影响应尽可能局限于变化点，不应导致全线路参数随机化。
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

## 3. 实现接口 (Proposed API)

```python
class ParameterMapper:
    def map(self, old_config: dict, old_params: Tensor, new_config: dict) -> Tensor:
        """
        根据配置差异，将旧参数映射到新维度空间。
        """
        pass
```

---

## 4. 下一步运行建议

- 在 `Phase 3` 实现 `IdentityMapper` 作为基准。
- 为 `ADAPT-VQE` 提供专门的 `GradientPriorityMapper`。
