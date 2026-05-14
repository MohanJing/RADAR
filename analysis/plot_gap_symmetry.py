import os
import numpy as np
import matplotlib.pyplot as plt

# ========================================================== #
# 请在这里填入你获得的目标值列表 (长度均为6)
# 分别对应对称性: [0%, 20%, 40%, 60%, 80%, 100%]
# ========================================================== #
# # 你可以修改这三个模型的名字
# MODEL_1_NAME = 'sinkhorn'
# MODEL_2_NAME = 'dim = 2'
# MODEL_3_NAME = 'dim = 3'

# # 三个模型的平均目标值（长度必须都是6）
# MODEL_1_RESULTS = [1.5778, 1.6618, 1.7912, 1.9782, 2.2880, 3.4444]
# MODEL_2_RESULTS = [1.5856, 1.6703, 1.8002, 1.9861, 2.2897, 3.3281]
# MODEL_3_RESULTS = [1.5922, 1.6800, 1.8132, 2.0038, 2.3169, 3.3047]

# 你可以修改这三个模型的名字
MODEL_1_NAME = 'sinkhorn'
MODEL_2_NAME = 'sinkhorn half mix train'
MODEL_3_NAME = 'sinkhorn full mix train'

# 三个模型的平均目标值（长度必须都是6）
MODEL_1_RESULTS = [1.5778, 1.6618, 1.7912, 1.9782, 2.2880, 3.4444]
MODEL_2_RESULTS = [1.6021, 1.6877, 1.8185, 2.0048, 2.3071, 3.0596]
MODEL_3_RESULTS = [3.0727, 2.8359, 2.8405, 2.9214, 3.0262, 3.1684]

LKH_RESULTS = [1.5670, 1.6502, 1.7766, 1.9563, 2.2428, 3.0125]  # LKH平均目标值

MODEL_SERIES = [
    {'name': MODEL_1_NAME, 'results': MODEL_1_RESULTS, 'color': 'tab:blue', 'marker': 'o'},
    {'name': MODEL_2_NAME, 'results': MODEL_2_RESULTS, 'color': 'tab:orange', 'marker': 's'},
    {'name': MODEL_3_NAME, 'results': MODEL_3_RESULTS, 'color': 'tab:green', 'marker': '^'},
]
# ========================================================== #

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
    labels = [f'{p}%' for p in p_values]

    # 基本长度检查，避免画图时报错
    if len(LKH_RESULTS) != len(p_values):
        raise ValueError(f'LKH_RESULTS长度必须为{len(p_values)}，当前为{len(LKH_RESULTS)}')
    for series in MODEL_SERIES:
        if len(series['results']) != len(p_values):
            raise ValueError(
                f"{series['name']} 的结果长度必须为{len(p_values)}，当前为{len(series['results'])}"
            )
    
    # 1. 计算不同模型在不同对称性程度下的 Gap
    # Gap = (Model - LKH) / LKH * 100%
    model_gaps = {}
    for series in MODEL_SERIES:
        gaps = [(m - l) / l * 100.0 for m, l in zip(series['results'], LKH_RESULTS)]
        model_gaps[series['name']] = gaps
    
    # 2. 读取 ATSP100 数据并计算相对 Frobenius 误差 (RFE)
    base_dir = '/data/jinmohan/RADAR/atsp/dataset/dataset_10k'
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
            
    # 3. 绘制折线图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制三个模型的 Gap vs 对称程度
    for series in MODEL_SERIES:
        ax1.plot(
            p_values,
            model_gaps[series['name']],
            marker=series['marker'],
            linestyle='-',
            color=series['color'],
            linewidth=2,
            markersize=8,
            label=series['name'],
        )
    ax1.set_xlabel('Degree of Symmetry', fontsize=12)
    ax1.set_ylabel('Gap to LKH (%)', fontsize=12)
    ax1.set_xticks(p_values)
    ax1.set_xticklabels(labels)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    
    # 设置上方刻度 (双X轴) 用于显示相对 Frobenius 误差
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(p_values)
    ax2.set_xticklabels([f'{rfe:.2f}' for rfe in rfes])
    ax2.set_xlabel('Relative Frobenius Error', fontsize=12, color='g')
    ax2.tick_params(axis='x', labelcolor='g')
    
    plt.title('Performance Gap vs Symmetry Degree (ATSP-100)', pad=20, fontsize=14)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(os.path.dirname(__file__), 'gap_vs_symmetry_train.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved successfully to: {save_path}")
    
    # 输出结果供检查
    print("\n========= Summary =========")
    for i, p in enumerate(p_values):
        gap_parts = [f"{series['name']}: {model_gaps[series['name']][i]:5.2f}%" for series in MODEL_SERIES]
        print(f"Symmetry {p:3d}% | {' | '.join(gap_parts)} | RFE: {rfes[i]:.4f}")

if __name__ == '__main__':
    main()
