import os
import numpy as np

def calculate_relative_frobenius_error(data):
    """
    计算相对 Frobenius 误差: ||M - M^T||_F / ||M||_F
    """
    # data shape: (batch_size, num_nodes, num_nodes)
    data_t = data.transpose(0, 2, 1) # 对每个矩阵求转置
    
    # 按照 frobenius norm公式计算
    diff_norm = np.linalg.norm(data - data_t, axis=(1, 2))
    mat_norm = np.linalg.norm(data, axis=(1, 2))
    
    # 防止分母为0
    rfe = diff_norm / (mat_norm + 1e-9)
    return np.mean(rfe)

def main():
    p_values = [0, 20, 40, 60, 80, 100]
    
    # 读取 ATSP100 数据并计算相对 Frobenius 误差 (RFE)
    base_dir = '/data/jinmohan/NCOdata/atsp_1k'
    rfes = []
    
    for p in p_values:
        filepath = os.path.join(base_dir, f'sampled_symmetry_p{p}', 'ATSP100_10k.npz')
        if os.path.exists(filepath):
            print(f"Loading {filepath} to calculate RFE...")
            data = np.load(filepath)['data']
            rfe = calculate_relative_frobenius_error(data)
            rfes.append(rfe)
        else:
            print(f"Warning: File not found {filepath} (Using 0.0 as placeholder)")
            rfes.append(0.0)
    print("RFE values for each symmetry level:", rfes)