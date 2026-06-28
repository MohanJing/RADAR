import numpy as np
import matplotlib.pyplot as plt
import os

def _load_result_costs(result_path):
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"找不到结果文件: {result_path}")
    result_data = np.load(result_path)
    if 'cost' not in result_data:
        raise KeyError(f"结果文件缺少 'cost' 字段: {result_path}")
    return result_data['cost']


def _print_stats(gaps, label):
    total_instances = len(gaps)
    mean_gap = np.mean(gaps)
    median_gap = np.median(gaps)
    max_gap = np.max(gaps)
    optimal_hits = np.sum(gaps <= 1e-5)
    optimal_rate = (optimal_hits / total_instances) * 100

    print(f"=== {label} 统计分析结果 ({total_instances} 个实例) ===")
    print(f"平均 Gap (Mean): {mean_gap:.4f}%")
    print(f"中位数 Gap (Median): {median_gap:.4f}%")
    print(f"最大 Gap (Max): {max_gap:.4f}%")
    print(f"最优解命中率 (Optimal Rate): {optimal_hits}/{total_instances} ({optimal_rate:.2f}%)")

    return {
        'total': total_instances,
        'mean': mean_gap,
        'median': median_gap,
        'max': max_gap,
        'optimal_rate': optimal_rate,
    }


def _plot_max_gap_marker(max_gap, bin_edges, percentages, color, label, y_offset=0):
    # 取包含 max gap 的 bin 百分比作为竖线高度
    bin_idx = np.searchsorted(bin_edges, max_gap, side='right') - 1
    bin_idx = int(np.clip(bin_idx, 0, len(percentages) - 1))
    y_at_max = percentages[bin_idx]

    plt.vlines(max_gap, 0, y_at_max, colors=color, linestyles='dashed', linewidth=1.2, alpha=0.9)
    plt.scatter([max_gap], [0], color=color, s=24, zorder=6)
    plt.annotate(
        f"{label} Max: {max_gap:.2f}%",
        xy=(max_gap, 0),
        xytext=(0, 8 + y_offset),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=9,
        color=color,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=color),
    )


def analyze_and_plot_gap(
    drl_result_path,
    baseline_result_path,
    save_fig_path="gap_distribution",
    problem="Unknown",
    drl_result_path_2=None,
    label_1="Solution A",
    label_2="Solution B",
    color_1="#4C72B0",
    color_2="#DD8452",
):
    # 1. 数据校验与加载
    if not os.path.exists(baseline_result_path):
        raise FileNotFoundError(f"找不到 Baseline 基准文件: {baseline_result_path}")

    baseline_data = np.load(baseline_result_path, allow_pickle=True).item()
    baseline_costs = baseline_data['costs']

    drl_costs_1 = _load_result_costs(drl_result_path)
    if len(drl_costs_1) != len(baseline_costs):
        raise ValueError("第一组解的实例数量与基准实例数量不一致")

    drl_costs_2 = None
    if drl_result_path_2 is not None:
        drl_costs_2 = _load_result_costs(drl_result_path_2)
        if len(drl_costs_2) != len(baseline_costs):
            raise ValueError("第二组解的实例数量与基准实例数量不一致")

    # 2. 核心计算：每个实例的 Gap (%)
    gaps_1 = (drl_costs_1 - baseline_costs) / baseline_costs * 100
    if not np.all(np.isfinite(gaps_1)):
        raise ValueError("第一组解中存在 NaN/Inf gap，请检查数据")

    gaps_2 = None
    if drl_costs_2 is not None:
        gaps_2 = (drl_costs_2 - baseline_costs) / baseline_costs * 100
        if not np.all(np.isfinite(gaps_2)):
            raise ValueError("第二组解中存在 NaN/Inf gap，请检查数据")

    # 3. 统计输出
    stats_1 = _print_stats(gaps_1, label_1)
    stats_2 = _print_stats(gaps_2, label_2) if gaps_2 is not None else None

    # 4. 统一分箱并截断 x 轴：按全局 max gap 向上取整
    all_min_gap = min(float(np.min(gaps_1)), 0.0)
    all_max_gap = float(np.max(gaps_1))
    if gaps_2 is not None:
        all_min_gap = min(all_min_gap, float(np.min(gaps_2)))
        all_max_gap = max(all_max_gap, float(np.max(gaps_2)))

    x_max = float(np.ceil(all_max_gap))
    if x_max <= all_min_gap:
        x_max = all_min_gap + 1.0

    bins = np.linspace(all_min_gap, x_max, 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    counts_1, bin_edges = np.histogram(gaps_1, bins=bins)
    percentages_1 = counts_1 / len(gaps_1) * 100

    plt.figure(figsize=(10, 6), dpi=150)

    plt.plot(
        bin_centers,
        percentages_1,
        marker='o',
        markersize=3,
        linewidth=1.8,
        color=color_1,
        alpha=0.95,
        label=label_1,
    )
    # 暂时关闭折线上的 max gap 标注（保留代码，便于后续恢复）
    # _plot_max_gap_marker(stats_1['max'], bin_edges, percentages_1, color_1, label_1, y_offset=0)

    if gaps_2 is not None:
        counts_2, _ = np.histogram(gaps_2, bins=bins)
        percentages_2 = counts_2 / len(gaps_2) * 100
        plt.plot(
            bin_centers,
            percentages_2,
            marker='s',
            markersize=3,
            linewidth=1.8,
            color=color_2,
            alpha=0.95,
            label=label_2,
        )
        assert stats_2 is not None
        # 暂时关闭折线上的 max gap 标注（保留代码，便于后续恢复）
        # _plot_max_gap_marker(stats_2['max'], bin_edges, percentages_2, color_2, label_2, y_offset=12)

    # 5. 图表修饰
    # plt.title(f"Distribution of Optimality Gap on {problem}", fontsize=14, fontweight='bold')
    plt.xlabel("Optimality Gap (%)", fontsize=12)
    plt.ylabel("Percentage of Instances (%)", fontsize=12)
    plt.xlim(all_min_gap, x_max)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if gaps_2 is not None:
        plt.legend(loc='upper right', fontsize=10, framealpha=0.9, bbox_to_anchor=(0.985, 0.985))

    stats_lines = [
        f"{label_1} Mean: {stats_1['mean']:.2f}%",
        f"{label_1} Median: {stats_1['median']:.2f}%",
        f"{label_1} Max: {stats_1['max']:.2f}%",
    ]
    if stats_2 is not None:
        stats_lines.extend(
            [
                f"{label_2} Mean: {stats_2['mean']:.2f}%",
                f"{label_2} Median: {stats_2['median']:.2f}%",
                f"{label_2} Max: {stats_2['max']:.2f}%",
            ]
        )

    stats_box_y = 0.98 if gaps_2 is None else 0.72

    plt.text(
        0.98,
        stats_box_y,
        "\n".join(stats_lines),
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
    )

    plt.tight_layout()
    plt.savefig(f"{save_fig_path}_{problem}.png")
    print(f"Gap 分布折线图已保存至: {save_fig_path}_{problem}.png")
    plt.show()

# 执行入口
if __name__ == "__main__":
    # 请根据你实际的文件结构调整路径
    DRL_FILE_1 = "/data/jinmohan/RADAR/atsp/result/answer/res_radar_official_checkpoint_1000.npz"
    # 设为 None 时仅绘制一条折线
    DRL_FILE_2 = "/data/jinmohan/RADAR/atsp/result/answer/res_zeroInit_w_edgeValue_w_bias_1000.npz"
    BASELINE_FILE = "/data/jinmohan/lkh/lkh_atsp500_results.npy"

    analyze_and_plot_gap(
        DRL_FILE_1,
        BASELINE_FILE,
        problem="ATSP1000",
        drl_result_path_2=DRL_FILE_2,
        label_1="RADAR",
        label_2="RADAR+edge",
    )