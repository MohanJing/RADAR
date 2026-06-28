#!/usr/bin/env python3
"""
Compare radar_official_checkpoint across different k values on the same TSP instances.
Computes the gap with LKH results and saves a table:
  - 1000 rows (instances)
  - 15 columns: 14 k values (gap %) + best_k (the k with the smallest absolute gap)

Output files:
  - k_comparison_gaps.csv       : gap (%) for each k per instance
  - k_comparison_costs.csv      : raw cost for each k per instance (for reference)
  - k_comparison_summary.csv    : per-k aggregate statistics
"""

import numpy as np
import pandas as pd
from pathlib import Path

node_cnt = 1000

# ── Paths ──────────────────────────────────────────────────────────────────
radar_dir = Path(f"/data/jinmohan/RADAR/atsp/result/answer/sampled_symmetry/p0/{node_cnt}")
lkh_path = Path(f"/data/jinmohan/lkh/result_1k/sampled_symmetry_p0/lkh_ATSP{node_cnt}_results.npy")
output_dir = Path(f"/data/jinmohan/RADAR/analysis/compare_k/{node_cnt}")

# ── Load LKH costs ─────────────────────────────────────────────────────────
lkh_data = np.load(lkh_path, allow_pickle=True).item()
lkh_costs = lkh_data["costs"]  # shape (1000,)
print(f"Loaded LKH costs: {lkh_costs.shape}, mean={lkh_costs.mean():.6f}")

# ── Load radar costs for each k ────────────────────────────────────────────
k_files = sorted(radar_dir.glob("radar_official_checkpoint_k=*.npz"))
k_values = []
all_costs = {}

for f in k_files:
    k_str = f.stem.split("k=")[-1]
    k = int(k_str)
    k_values.append(k)
    data = np.load(f)
    all_costs[k] = data["cost"].copy()
    data.close()

k_values = sorted(k_values)
print(f"Loaded radar costs for {len(k_values)} k values: {k_values}")
for k in k_values:
    print(f"  k={k:>4d}: mean cost = {all_costs[k].mean():.6f}")

# ── Compute gaps ───────────────────────────────────────────────────────────
# gap (%) = (radar_cost - lkh_cost) / lkh_cost * 100
gaps = {}
for k in k_values:
    gaps[k] = (all_costs[k] - lkh_costs) / lkh_costs * 100

# ── Build gap DataFrame ────────────────────────────────────────────────────
df_gaps = pd.DataFrame(gaps, columns=k_values)
df_gaps.index.name = "instance"

# ── Determine best k (lowest radar cost; ties prefer k=20) ─────────────────
best_k = []
for i in range(1000):
    min_cost = min(all_costs[k][i] for k in k_values)
    tied = [k for k in k_values if all_costs[k][i] == min_cost]
    # Tie-breaking: if k=20 is among the best, prefer it
    best = 20 if 20 in tied else tied[0]
    best_k.append(best)

# Count how many instances had ties involving k=20
tie_count_20 = sum(
    1 for i in range(1000)
    if min(all_costs[k][i] for k in k_values) == all_costs[20][i]
    and len([k for k in k_values if all_costs[k][i] == all_costs[20][i]]) > 1
)
print(f"\nTies resolved by k=20 preference: {tie_count_20} instances")

df_gaps["best_k"] = best_k

# Round for readability
df_gaps = df_gaps.round(4)

# ── Build cost DataFrame (reference) ───────────────────────────────────────
df_costs = pd.DataFrame(all_costs, columns=k_values)
df_costs.index.name = "instance"
df_costs["lkh_cost"] = lkh_costs
df_costs["best_k"] = best_k
df_costs = df_costs.round(6)

# ── Save ───────────────────────────────────────────────────────────────────
gap_path = output_dir / f"{node_cnt}k_comparison_gaps.csv"
cost_path = output_dir / f"{node_cnt}k_comparison_costs.csv"
df_gaps.to_csv(gap_path)
df_costs.to_csv(cost_path)
print(f"\nGap table saved to: {gap_path}  (shape: {df_gaps.shape})")
print(f"Cost table saved to: {cost_path}  (shape: {df_costs.shape})")

# ── Summary statistics ─────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("GAP (%) STATISTICS BY K")
print("=" * 100)
summary = df_gaps[k_values].describe().T
summary["best_k_count"] = [df_gaps["best_k"].value_counts().get(k, 0) for k in k_values]
summary["best_k_pct"] = summary["best_k_count"] / 10.0  # percent of 1000 instances
summary = summary.round(4)
print(summary.to_string())

summary_path = output_dir / f"{node_cnt}k_comparison_summary.csv"
summary.to_csv(summary_path)
print(f"\nSummary saved to: {summary_path}")

# ── Print best-k distribution ──────────────────────────────────────────────
print("\n" + "-" * 60)
print("BEST K DISTRIBUTION (k that achieves lowest cost per instance)")
print("-" * 60)
vk = df_gaps["best_k"].value_counts().sort_index()
for k, cnt in vk.items():
    bar = "█" * (cnt // 5)
    print(f"  k={k:>4d}: {cnt:>4d} instances ({cnt/10:.1f}%)  {bar}")

# ── Oracle comparison: k=20 for all vs. per-instance best_k ─────────────────
gap_all_k20 = np.mean([abs(gaps[20][i]) for i in range(1000)])
gap_per_best = np.mean([abs(gaps[best_k[i]][i]) for i in range(1000)])

print("\n" + "=" * 60)
print("ORACLE COMPARISON")
print("=" * 60)
print(f"  Mean |gap| when all instances use k=20:     {gap_all_k20:.4f}%")
print(f"  Mean cost when all instances use k=20:       {all_costs[20].mean():.6f}")
print(f"  Mean |gap| when each instance uses best_k:   {gap_per_best:.4f}%")
oracle_costs = np.array([all_costs[best_k[i]][i] for i in range(1000)])
print(f"  Mean cost when each instance uses best_k:     {oracle_costs.mean():.6f}")
print(f"  Gap reduction:                               {gap_all_k20 - gap_per_best:.4f}%")
print(f"  Relative improvement:                        {(gap_all_k20 - gap_per_best) / gap_all_k20 * 100:.2f}%")

print("\nDone.")
