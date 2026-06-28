import numpy as np
import pandas as pd
from scipy.stats import entropy, skew
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据加载模块 (需根据实际格式微调)
# ==========================================
def load_data(npz_path, csv_path, strict_best=False):
    """
    加载距离矩阵和对应的最优 K 值标签。

    Parameters
    ----------
    strict_best : bool
        如果为 True，则只保留最优 K 值**唯一**的实例（即只有 1 个 K
        值达到最低 cost），剔除多个 K 值打平手的模糊样本。
    """
    loaded = np.load(npz_path, allow_pickle=True)
    # 优先使用 'data' 键，否则使用第一个键
    if 'data' in loaded:
        matrices = loaded['data']
    else:
        key = loaded.files[0]
        matrices = loaded[key]

    if matrices.ndim != 3:
        raise ValueError(f"Expected 3D array (N, node, node), got shape {matrices.shape}")

    df = pd.read_csv(csv_path)

    if 'best_k' not in df.columns:
        raise ValueError(f"CSV must contain 'best_k' column, got: {list(df.columns)}")

    # --- Strict Best 过滤逻辑 ---
    if strict_best:
        # K 值列名：能转为数字且不是 instance / lkh_cost / best_k
        non_k_cols = {'instance', 'lkh_cost', 'best_k'}
        k_cols = [c for c in df.columns if c not in non_k_cols]

        costs = df[k_cols].values.astype(float)
        min_costs = costs.min(axis=1)
        tie_counts = (costs == min_costs[:, None]).sum(axis=1)

        strict_mask = tie_counts == 1
        n_total = len(df)
        n_strict = strict_mask.sum()
        print(f"Strict best 过滤: {n_strict}/{n_total} 个实例有唯一最优 K "
              f"(剔除了 {n_total - n_strict} 个有 tie 的样本)")

        df = df[strict_mask].reset_index(drop=True)
        matrices = matrices[strict_mask]

    labels = df['best_k'].values
    # 确保 labels 为整数类型（pandas 读取时可能为 object）
    labels = labels.astype(int)

    if len(matrices) != len(labels):
        raise ValueError(
            f"Mismatch: {len(matrices)} instances in npz but {len(labels)} in CSV"
        )

    return matrices, labels

# ==========================================
# 2. 特征工程模块 (第一性原理物理量提取)
# ==========================================
def extract_features(matrices):
    """
    计算输入矩阵数组 (B, N, N) 的 4 个核心特征
    """
    B, N, _ = matrices.shape
    features = np.zeros((B, 4))
    
    for i in range(B):
        D = matrices[i]
        
        # 为了防止极端值，设定一个温度系数 tau 为均值
        tau = np.mean(D) + 1e-6
        # 近似注意力得分矩阵 E
        E = np.exp(-D / tau)
        E = D

        # --- 特征 1：预演边缘差异度 (Pre-Marginal Discrepancy) ---
        row_sum = np.sum(E, axis=1)
        col_sum = np.sum(E, axis=0)
        # 使用最大绝对偏差作为震荡强度的代理
        features[i, 0] = np.max(np.abs(row_sum - col_sum))
        
        # --- 特征 2：行列信息熵增益差 (Row-Column Entropy Delta) ---
        # 计算行概率分布和列概率分布
        P_row = E / (row_sum[:, None] + 1e-8)
        P_col = E / (col_sum[None, :] + 1e-8)
        
        # 分别计算每行和每列的香农熵，并取平均
        H_row_mean = np.mean(entropy(P_row, axis=1))
        H_col_mean = np.mean(entropy(P_col, axis=0))
        features[i, 1] = H_row_mean - H_col_mean
        
        # --- 特征 3：极端单向节点占比 (Extreme Unidirectional Node Ratio) ---
        D_row_mean = np.mean(D, axis=1)
        D_col_mean = np.mean(D, axis=0) + 1e-8
        ratio = D_row_mean / D_col_mean
        # 统计异常节点（出入度代价悬殊 > 2倍 或 < 0.5倍）的比例
        extreme_nodes = np.sum((ratio > 2.0) | (ratio < 0.5))
        features[i, 2] = extreme_nodes / N
        
        # --- 特征 4：矩阵的非对称偏度 (Skewness of Asymmetry) ---
        A = D - D.T
        # 提取非对称残差矩阵中严格大于 0 的元素（代表 $i \to j$ 比 $j \to i$ 难的部分）
        A_pos = A[A > 0]
        if len(A_pos) > 2:
            features[i, 3] = skew(A_pos)
        else:
            features[i, 3] = 0.0

    return pd.DataFrame(features, columns=[
        'F1_Marginal_Discrepancy', 
        'F2_Entropy_Delta', 
        'F3_Extreme_Node_Ratio', 
        'F4_Asymmetry_Skewness'
    ])

# ==========================================
# 3. 训练与评估模块
# ==========================================
def train_and_evaluate(X, y):
    """
    训练决策树并输出特征重要性
    """
    # 划分训练集和测试集 (80% 训练，20% 验证)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 限制树的最大深度，防止过拟合，并保持极高的可解释性
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # 预测并计算准确率
    y_pred = dt_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 基线准确率 (总是预测占比最多的类别)
    print("\n【类别分布】")
    class_dist = pd.Series(y).value_counts().sort_index()
    for k, v in class_dist.items():
        print(f"  best_k = {k:>4}: {v:>4} samples")
    print(f"  共 {len(class_dist)} 个类别, 总 {len(y)} 个样本")

    majority_class = pd.Series(y_train).mode().iloc[0]
    baseline_acc = np.mean(y_test == majority_class)
    
    print("="*40)
    print("【准确率评估】")
    print(f"多数类基准准确率 (Baseline): {baseline_acc:.4f}")
    print(f"决策树模型准确率 (Model)   : {acc:.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("="*40)
    
    # 提取特征重要性
    importances = dt_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print("【特征重要性 (Gini Importance)】")
    print(importance_df.to_string(index=False))
    
    # 可视化决策树
    plt.figure(figsize=(16, 10))
    plot_tree(dt_model, feature_names=feature_names, class_names=[str(c) for c in dt_model.classes_], 
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree for Optimal K Prediction")
    plt.savefig("decision_tree_visualization.png", dpi=300)
    print("\n决策树结构图已保存为 'decision_tree_visualization.png'")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    npz_file = "/data/jinmohan/NCOdata/atsp_1k/sampled_symmetry_p0/ATSP100.npz"
    csv_file = "/data/jinmohan/RADAR/analysis/compare_k/100/100k_comparison_costs.csv"

    # 开关：只保留有严格唯一最优 K 的实例（剔除多 K 打平手的模糊样本）
    STRICT_BEST = True

    try:
        # 1. 加载数据（优先使用真实数据，不存在则回退到模拟数据）
        matrices, labels = load_data(npz_file, csv_file, strict_best=STRICT_BEST)
        print(f"成功加载真实数据: {matrices.shape[0]} 条实例, "
              f"矩阵大小 {matrices.shape[1]}×{matrices.shape[2]}")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"真实数据加载失败 ({e})，使用随机生成的模拟数据进行流程演示...")
        matrices = np.random.rand(1000, 100, 100) * 100
        labels = np.random.choice([10, 20, 30, 40], size=1000, p=[0.15, 0.46, 0.20, 0.19])

    # 2. 提取特征
    print("正在进行物理特征提取 (O(N^2) 复杂度)...")
    X = extract_features(matrices)

    # 3. 训练与评估
    train_and_evaluate(X, labels)