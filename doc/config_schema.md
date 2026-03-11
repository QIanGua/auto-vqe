# Agent-VQE 配置规范 (Config Schema)

> **版本**: 1.0 (P1)  
> **状态**: 草案  
> **对象**: `AnsatzSpec`, `SearchSpaceSpec`, `OptimizerSpec`

为了减少手工 `dict` 带来的拼写错误与校验困难，本项目将逐步过渡到类型化配置系统。

---

## 1. Ansatz 配置 (`AnsatzSpec`)

描述一个完整的线路结构。

| 字段 | 类型 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| `name` | `str` | Ansatz 唯一标识 | `"tfim_hea_v1"` |
| `family` | `str` | 类别 (hea, uccsd, hva, ga, etc.) | `"hea"` |
| `n_qubits` | `int` | 量子比特数 | `4` |
| `config` | `dict` | 结构化具体参数 | `{ "layers": 2, ... }` |
| `metadata` | `dict` | 额外元信息 | `{ "search_gen": 15 }` |

### `config` 内部规范 (HEA 示例)
- `layers`: 循环层数。
- `single_qubit_gates`: 单比特门列表，如 `["rx", "ry"]`。
- `two_qubit_gate`: 双比特门类型，如 `"cz"`。
- `entanglement`: 拓扑结构，如 `"full"`, `"linear"`, `"nearest_neighbor"`。

---

## 2. 搜索空间配置 (`SearchSpaceSpec`)

定义 `GASearchStrategy` 或 `GridSearch` 的探索范围。

```json
{
  "layers": [1, 2, 3, 4],
  "entanglement": ["linear", "full"],
  "single_qubit_gates_options": [
    ["rx", "ry"],
    ["rx", "ry", "rz"]
  ]
}
```

---

## 3. 优化器配置 (`OptimizerSpec`)

```json
{
  "method": "Adam",
  "lr": 0.01,
  "max_steps": 500,
  "tol": 1e-8,
  "scheduler": {
    "type": "ReduceLROnPlateau",
    "patience": 50
  }
}
```

---

## 4. 下一步计划 (Roadmap)

- [ ] 实现 `core/schemas.py`，使用 `pydantic.BaseModel` 进行运行时校验。
- [ ] 在 `circuit_factory.py` 中强制执行 schema 检查。
