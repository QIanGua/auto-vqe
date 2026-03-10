"""
environment.py: The Laws of Physics (Read-Only)
这里定义了客观现实：4量子比特的一维横场伊辛模型(TFIM)。
大自然法则不可更改，Agent 绝对不能修改此文件。
"""

import numpy as np

# 物理系统参数
N_QUBITS = 4
MAX_STEPS = 200      # 经典优化器的最大迭代步数
TIME_BUDGET = 60     # 每次实验的最大时间预算（秒）

def get_exact_gs_energy():
    r"""
    第一性原理：通过严格对角化(Exact Diagonalization)获取真实的基态能量。
    这是 AI 试图逼近的“绝对真理”。
    H = - \sum_{i} Z_i Z_{i+1} - \sum_{i} X_i
    """
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])

    def tensor_op(op, pos):
        res = 1
        for i in range(N_QUBITS):
            res = np.kron(res, op if i == pos else I)
        return res

    H = np.zeros((2**N_QUBITS, 2**N_QUBITS))
    
    # 相互作用项 (ZZ)
    for i in range(N_QUBITS - 1):
        H -= tensor_op(Z, i) @ tensor_op(Z, i+1)
    
    # 横场项 (X)
    for i in range(N_QUBITS):
        H -= tensor_op(X, i)
        
    eigenvalues = np.linalg.eigvalsh(H)
    return np.min(eigenvalues)

# 计算理论极限能量 (-4.7487 左右)
EXACT_ENERGY = get_exact_gs_energy()

if __name__ == "__main__":
    print(f"Objective Reality (Exact Ground State Energy): {EXACT_ENERGY:.6f}")