#!/usr/bin/env python3
"""
Compute skewness and kurtosis of pairwise distance distributions for 4 types of
TSP/ATSP distance matrices (N=100, 1000 samples each).

Datasets:
  1. atsp_tmat        — asymmetric TSP in TMAT format (symmetry p=0)
  2. tsp_tmat         — symmetric TSP in TMAT format (symmetry p=100, averaged)
  3. atsp_wo_ti       — asymmetric TSP without triangle inequality
  4. euc_tsp          — 2D Euclidean TSP

For each sample we extract all off-diagonal distances (100×99 = 9900 values)
and compute skewness and kurtosis, then summarise across the 1000 samples.
"""

import numpy as np
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
N = 100
NUM_SAMPLES = 1000

DATASETS = {
    "atsp_tmat": {
        "path": "/data/jinmohan/NCOdata/atsp_1k/sampled_symmetry_p0/ATSP100.npz",
        "desc": "ATSP (tmat, sym p=0)",
    },
    "tsp_tmat": {
        "path": "/data/jinmohan/NCOdata/atsp_1k/sampled_symmetry_p100/ATSP100.npz",
        "desc": "TSP (tmat, sym p=100, avg)",
    },
    "atsp_wo_ti": {
        "path": "/data/jinmohan/NCOdata/atsp_1k_wo_ti/ATSP100_wo_ti.npz",
        "desc": "ATSP without triangle inequality",
    },
    "euc_tsp": {
        "path": "/data/jinmohan/NCOdata/euc_tsp_1k/euc_TSP100.npz",
        "desc": "2D Euclidean TSP",
    },
}

OUT_DIR = Path("/data/jinmohan/RADAR/analysis/input")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ─────────────────────────────────────────────────────────────────
def extract_off_diagonal(dist_matrix: np.ndarray) -> np.ndarray:
    """Extract all off-diagonal elements from an (N, N) distance matrix."""
    n = dist_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return dist_matrix[mask]


def skewness(x: np.ndarray) -> float:
    """Sample skewness (bias-corrected Fisher–Pearson G1)."""
    n = len(x)
    mu = np.mean(x)
    m2 = np.mean((x - mu) ** 2)
    m3 = np.mean((x - mu) ** 3)
    if m2 <= 0:
        return float('nan')
    return float((n ** 2) / ((n - 1) * (n - 2)) * (m3 / (m2 ** 1.5)))


def kurtosis(x: np.ndarray) -> float:
    """Sample excess kurtosis (bias-corrected Fisher–Pearson G2)."""
    n = len(x)
    mu = np.mean(x)
    m2 = np.mean((x - mu) ** 2)
    m4 = np.mean((x - mu) ** 4)
    if m2 <= 0:
        return float('nan')
    g2 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * (m4 / (m2 ** 2)) \
         - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float(g2)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    results = {}
    sk_samples = {}
    kt_samples = {}

    print(f"Computing skewness & kurtosis for N={N}, {NUM_SAMPLES} samples each")
    print(f"Off-diagonal values per sample: {N * (N - 1)}")

    for key, info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {info['desc']}")
        print(f"  File: {info['path']}")

        data = np.load(info["path"])["data"]               # (1000, 100, 100)
        print(f"  Shape: {data.shape}, dtype: {data.dtype}")
        assert data.shape == (NUM_SAMPLES, N, N), f"Unexpected shape: {data.shape}"

        sk_list = np.empty(NUM_SAMPLES, dtype=np.float64)
        kt_list = np.empty(NUM_SAMPLES, dtype=np.float64)

        for i in range(NUM_SAMPLES):
            dists = extract_off_diagonal(data[i])
            sk_list[i] = skewness(dists)
            kt_list[i] = kurtosis(dists)

            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{NUM_SAMPLES} samples done")

        # summary statistics
        stats = {
            "desc": info["desc"],
            "skewness": {
                "mean":  float(np.mean(sk_list)),
                "std":   float(np.std(sk_list, ddof=1)),
                "min":   float(np.min(sk_list)),
                "max":   float(np.max(sk_list)),
                "median": float(np.median(sk_list)),
            },
            "kurtosis": {
                "mean":  float(np.mean(kt_list)),
                "std":   float(np.std(kt_list, ddof=1)),
                "min":   float(np.min(kt_list)),
                "max":   float(np.max(kt_list)),
                "median": float(np.median(kt_list)),
            },
        }
        results[key] = stats
        sk_samples[key] = sk_list
        kt_samples[key] = kt_list

        sk = stats["skewness"]
        kt = stats["kurtosis"]
        print(f"  Skewness  — mean: {sk['mean']:.4f}, std: {sk['std']:.4f}, "
              f"median: {sk['median']:.4f}, range: [{sk['min']:.4f}, {sk['max']:.4f}]")
        print(f"  Kurtosis  — mean: {kt['mean']:.4f}, std: {kt['std']:.4f}, "
              f"median: {kt['median']:.4f}, range: [{kt['min']:.4f}, {kt['max']:.4f}]")

    # ── summary table ───────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{'Dataset':<38} {'Skewness (μ±σ)':<30} {'Excess Kurtosis (μ±σ)':<30}")
    print("-" * 90)
    for key in DATASETS:
        sk = results[key]["skewness"]
        kt = results[key]["kurtosis"]
        print(f"{results[key]['desc']:<38} {sk['mean']:>9.4f} ± {sk['std']:<9.4f}  "
              f"{kt['mean']:>9.4f} ± {kt['std']:<9.4f}")

    # ── save summary ───────────────────────────────────────────────────────
    summary_path = OUT_DIR / f"skew_kurt_N{N}_summary.npz"
    save_dict = {}
    for key in DATASETS:
        save_dict[f"{key}_skewness"] = np.array([
            results[key]["skewness"][k] for k in ("mean", "std", "min", "max", "median")
        ])
        save_dict[f"{key}_kurtosis"] = np.array([
            results[key]["kurtosis"][k] for k in ("mean", "std", "min", "max", "median")
        ])
    np.savez_compressed(summary_path, **save_dict)
    print(f"\nSummary saved to: {summary_path}")

    # ── save per-sample arrays (for box-plots / distribution analysis) ──────
    persample_path = OUT_DIR / f"skew_kurt_N{N}_persample.npz"
    ps_save = {}
    for key in DATASETS:
        ps_save[f"{key}_skewness"] = sk_samples[key]
        ps_save[f"{key}_kurtosis"] = kt_samples[key]
    np.savez_compressed(persample_path, **ps_save)
    print(f"Per-sample data saved to: {persample_path}")


if __name__ == "__main__":
    main()
