import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Find top n instances with the largest optimality gap (Model vs LKH).")
    parser.add_argument('--lkh_path', type=str, 
                        default='/data/jinmohan/lkh/result/sampled_symmetry_p100/lkh_ATSP100_10k_results.npy', 
                        help='Path to LKH results (.npy)')
    parser.add_argument('--model_path', type=str, 
                        default='/data/jinmohan/RADAR/atsp/result/answer/sampled_symmetry/res_radar_official_checkpoint_100_p100.npz', 
                        help='Path to Model results (.npz)')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top instances to retrieve')
    args = parser.parse_args()

    # 1. 读取数据
    print(f"Loading LKH results from: {args.lkh_path}")
    lkh_data = np.load(args.lkh_path, allow_pickle=True)
    
    # 如果数据是标量字典形式，从中提取 costs
    if lkh_data.shape == ():
        lkh_data = lkh_data.item()
        
    if isinstance(lkh_data, dict) and 'costs' in lkh_data:
        lkh_costs = lkh_data['costs']
    else:
        lkh_costs = lkh_data
        
    print(f"Loading Model results from: {args.model_path}")
    model_data = np.load(args.model_path)
    
    # 从 npz 中提取 cost
    if 'cost' in model_data:
        model_costs = model_data['cost']
    else:
        raise KeyError(f"'cost' not found in {args.model_path}. Available keys: {list(model_data.keys())}")

    # 2. 检查长度一致性
    assert len(lkh_costs) == len(model_costs), f"Data length mismatch: LKH ({len(lkh_costs)}) vs Model ({len(model_costs)})"
    total_instances = len(lkh_costs)
    print(f"Successfully loaded {total_instances} instances.\n")

    # 3. 逐条计算 Gap 
    # 计算公式: Gap = (Model_Cost - LKH_Cost) / LKH_Cost * 100%
    # 为了防止除以0的情况，加入极小值保护
    lkh_costs_safe = np.where(lkh_costs == 0, 1e-9, lkh_costs)
    gaps = (model_costs - lkh_costs) / lkh_costs_safe * 100.0

    # 4. 获取 Gap 最大的 top n 的实例下标
    # np.argsort默认是升序排列，[-args.top_n:]取出最后n个（即最大的n个），[::-1]将其翻转为降序
    top_n_indices = np.argsort(gaps)[-args.top_n:][::-1]

    # 5. 输出结果
    print(f"{'='*50}")
    print(f"Top {args.top_n} instances with the LARGEST Gap")
    print(f"{'='*50}")
    for i, idx in enumerate(top_n_indices):
        m_cost = model_costs[idx]
        l_cost = lkh_costs[idx]
        gap_val = gaps[idx]
        print(f"Rank {i+1:2d} | Instance Index (0-based): {idx:5d} | Model Cost: {m_cost:10.4f} | LKH Cost: {l_cost:10.4f} | Gap: {gap_val:8.2f}%")

if __name__ == '__main__':
    main()
