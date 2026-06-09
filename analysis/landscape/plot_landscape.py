import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import random

def calculate_tour_length(tour, dist_matrix):
    """Calculate the total Euclidean distance of a tour."""
    length = 0.0
    num_nodes = len(tour)
    for i in range(num_nodes):
        length += dist_matrix[tour[i], tour[(i+1) % num_nodes]]
    return length

def two_opt_swap(tour):
    """Perform a random 2-opt swap on the tour."""
    new_tour = tour.copy()
    num_nodes = len(tour)
    i, j = sorted(random.sample(range(num_nodes), 2))
    new_tour[i:j+1] = list(reversed(new_tour[i:j+1]))
    return new_tour

def get_edges(tour):
    """Get a set of undirected edges from a tour."""
    edges = set()
    num_nodes = len(tour)
    for i in range(num_nodes):
        u, v = tour[i], tour[(i+1) % num_nodes]
        edges.add((min(u, v), max(u, v)))
    return edges

def calculate_edge_distance(tour1, tour2):
    """Calculate edge difference distance between two tours. 
    It returns the number of edges in tour1 that are not in tour2."""
    edges1 = get_edges(tour1)
    edges2 = get_edges(tour2)
    return len(edges1) - len(edges1.intersection(edges2))

def main():
    # Fix random seeds for reproducibility
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)

    node_cnt = 1000

    print("Loading data...")
    # 1. Load Data
    # dist_data = np.load(f'/data/jinmohan/NCOdata/euc_tsp_1k/euc_TSP{node_cnt}.npz')
    # dist_data = np.load(f'/data/jinmohan/NCOdata/atsp_1k/sampled_symmetry_p0/ATSP{node_cnt}.npz')
    dist_data = np.load(f'/data/jinmohan/NCOdata/atsp_1k/sampled_symmetry_p100/ATSP{node_cnt}.npz')
    # dist_data = np.load(f'/data/jinmohan/NCOdata/atsp_1k_wo_ti/ATSP{node_cnt}_wo_ti.npz')


    # lkh_data = np.load(f'/data/jinmohan/lkh/result_1k_euc/lkh_euc_TSP{node_cnt}_results.npy', allow_pickle=True).item()
    # lkh_data = np.load(f'/data/jinmohan/lkh/result_1k/sampled_symmetry_p0/lkh_ATSP{node_cnt}_results.npy', allow_pickle=True).item()
    lkh_data = np.load(f'/data/jinmohan/lkh/result_1k/sampled_symmetry_p100/lkh_ATSP{node_cnt}_results.npy', allow_pickle=True).item()
    # lkh_data = np.load(f'/data/jinmohan/lkh/result_1k_wo_ti/lkh_ATSP{node_cnt}_wo_ti_results.npy', allow_pickle=True).item()

    dist_matrix = dist_data['data'][0] / 1e6 # Shape (100, 100), divided by 1e6 as per data format

    opt_tour = np.array(lkh_data['routes'][0]) - 1 # 1-indexed to 0-indexed
    opt_cost = lkh_data['costs'][0]
    
    print(f"Optimal tour length (LKH): {opt_cost:.4f}")

    # 2. Sample Solution Space
    print("Sampling solution space via 2-opt perturbations...")
    samples = []
    fitness = []
    
    # Add optimal tour first (Index 0)
    samples.append(opt_tour)
    fitness.append(calculate_tour_length(opt_tour, dist_matrix))
    
    num_random_walks = 100
    steps_per_walk = 10
    
    # Generate neighbor tours
    for _ in range(num_random_walks):
        curr_tour = opt_tour.copy()
        for step in range(steps_per_walk):
            curr_tour = two_opt_swap(curr_tour)
            samples.append(curr_tour)
            fitness.append(calculate_tour_length(curr_tour, dist_matrix))
            
    # Add some completely random tours to see the farther landscape
    # for _ in range(100):
    #     rand_tour = list(range(100))
    #     random.shuffle(rand_tour)
        # samples.append(np.array(rand_tour))
        # fitness.append(calculate_tour_length(rand_tour, dist_matrix))

    num_samples = len(samples)
    print(f"Total sampled tours: {num_samples}")

    # 3. Calculate Structural Distances
    print("Building pairwise distance matrix...")
    dist_mat = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        # We can optimize this by storing edge sets
        pass
    
    edge_sets = [get_edges(t) for t in samples]
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            # symmetric edge difference
            diff = len(edge_sets[i]) - len(edge_sets[i].intersection(edge_sets[j]))
            dist_mat[i, j] = diff
            dist_mat[j, i] = diff

    # 4. Dimensionality Reduction
    print("Applying MDS for dimensionality reduction (this may take a moment)...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
    coords_2d = mds.fit_transform(dist_mat)
    
    X = coords_2d[:, 0]
    Y = coords_2d[:, 1]
    Z = np.array(fitness)

    # --- Control variable: 'tour_length' or 'gap' ---
    color_mode = 'gap'  # Change to 'gap' to color by gap percentage instead of raw tour length

    # Compute Gap(%) relative to optimal
    optimal_length = Z[0]
    gap_percent = 100.0 * (Z - optimal_length) / optimal_length

    # 5. 2D Visualization
    print("Generating 2D Visualization...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    if color_mode == 'gap':
        # Cap gap at 200% for color mapping; values >200% use a uniform "over" color
        gap_clipped = np.clip(gap_percent, 0, 200)
        # Dynamically set vmax: scale to actual data range, capped at 200%
        vmax = min(200, gap_clipped.max())
        has_over = gap_percent.max() > 200
        cmap = plt.cm.viridis
        cmap.set_over('darkred')
        scatter = ax.scatter(X, Y, c=gap_clipped, cmap=cmap, vmin=0, vmax=vmax,
                             s=30, alpha=0.8, edgecolor='none')
        cbar = fig.colorbar(scatter, ax=ax, label='Gap (%)',
                            extend='max' if has_over else 'neither')
        print(f"Gap stats: min={gap_percent.min():.2f}%, max={gap_percent.max():.2f}%, "
              f"mean={gap_percent.mean():.2f}%, vmax={vmax:.2f}")
    else:
        scatter = ax.scatter(X, Y, c=Z, cmap='viridis', s=30, alpha=0.8, edgecolor='none')
        fig.colorbar(scatter, ax=ax, label='Tour Length (Fitness)')

    # Highlight the optimal solution
    ax.scatter(X[0], Y[0], color='red', s=150, marker='*', label=f'Optimal (Cost: {Z[0]:.2f})')

    ax.set_title(f"2D Fitness Landscape of {node_cnt} ({'Gap' if color_mode == 'gap' else 'Tour Length'})")
    ax.set_xlabel("Structural Dim 1 (MDS)")
    ax.set_ylabel("Structural Dim 2 (MDS)")

    ax.legend()

    # out_path = f'/data/jinmohan/RADAR/analysis/landscape/euc_tsp/{node_cnt}_{color_mode}.png'
    # out_path = f'/data/jinmohan/RADAR/analysis/landscape/atsp_p0/{node_cnt}_{color_mode}.png'
    out_path = f'/data/jinmohan/RADAR/analysis/landscape/atsp_p100/{node_cnt}_{color_mode}.png'
    # out_path = f'/data/jinmohan/RADAR/analysis/landscape/atsp_wo_ti/{node_cnt}_{color_mode}.png'

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {out_path}")

if __name__ == '__main__':
    main()