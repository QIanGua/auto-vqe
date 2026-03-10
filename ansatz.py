"""
ansatz.py: The Conjecture (Mutable)
这是 AI (Agent) 进行科学探索的实验室。
Agent 可以随意更改线路结构（Ansatz）、门的类型、连接拓扑、甚至优化器逻辑。
"""

# 1. 消除第三方库的无谓告警 (使用 contextlib 重定向以彻底沉默导入时的 print)
import os
import warnings
import sys
from contextlib import redirect_stdout, redirect_stderr

with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull), redirect_stderr(fnull):
        import tensorcircuit as tc
        import torch

import time
from environment import N_QUBITS, MAX_STEPS, TIME_BUDGET, EXACT_ENERGY

warnings.filterwarnings("ignore", category=SyntaxWarning)  # 忽略 tensorcircuit 内部的 LaTeX 告警

# 2. 设置随机种子以保证实验可重复性
torch.manual_seed(42)

# 设置后端为 pytorch
tc.set_backend("pytorch")

def create_circuit(params):
    """
    【AI 的假设空间】：这是一个极其基础的基线(Baseline) Ansatz。
    Agent 应该在这里尝试 Hardware-Efficient Ansatz, UCCSD启发式, 甚至是随机拓扑。
    """
    c = tc.Circuit(N_QUBITS)
    idx = 0
    
    # 第一层：局部旋转
    for i in range(N_QUBITS):
        c.ry(i, theta=params[idx])
        idx += 1
        
    # 第二层：线性纠缠
    for i in range(N_QUBITS - 1):
        c.cnot(i, i+1)
        
    # 第三层：局部旋转
    for i in range(N_QUBITS):
        c.rx(i, theta=params[idx])
        idx += 1
        
    return c, idx

def compute_energy(params):
    r"""
    计算哈密顿量的期望值 <ψ(θ)|H|ψ(θ)>
    H = - \sum Z_i Z_{i+1} - \sum X_i
    """
    c, _ = create_circuit(params)
    energy = 0.0
    
    # - Z_i Z_{i+1} 测量
    for i in range(N_QUBITS - 1):
        energy += -c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [i+1]])
        
    # - X_i 测量
    for i in range(N_QUBITS):
        energy += -c.expectation([tc.gates.x(), [i]])
        
    return tc.backend.real(energy)

def train():
    """
    执行“反驳”过程（Refutation）：用梯度下降检验并优化假设。
    """
    # 初始化一个虚拟线路以获取所需参数数量
    _, num_params = create_circuit(torch.zeros(1000))
    
    # 初始化参数
    params = torch.randn(num_params).detach() * 0.1
    params.requires_grad_(True)
    
    # 使用 Adam 优化器
    optimizer = torch.optim.Adam([params], lr=0.05)
    
    start_time = time.time()
    best_energy = float('inf')
    
    # 运行更多步以确保收敛
    for step in range(1000):
        optimizer.zero_grad()
        energy = compute_energy(params)
        energy.backward()
        optimizer.step()
        
        current_energy = energy.item()
        if current_energy < best_energy:
            best_energy = current_energy
            
        if time.time() - start_time > TIME_BUDGET:
            break

    # === 输出区 (必须严格保持此格式以供 Agent 提取) ===
    print("---")
    print(f"val_energy:       {float(best_energy):.6f}")
    print(f"exact_energy:     {EXACT_ENERGY:.6f}")
    print(f"energy_error:     {float(best_energy) - EXACT_ENERGY:.6f}")
    print(f"num_params:       {num_params}")
    print(f"training_seconds: {time.time() - start_time:.2f}")

if __name__ == "__main__":
    train()