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
    print("Loading data...")
    # 1. Load Data
    dist_data = np.load('/data/jinmohan/NCOdata/euc_tsp_1k/euc_TSP100.npz')
    # dist_data = np.load('/data/jinmohan/NCOdata/atsp_1k/sampled_symmetry_p0/ATSP100.npz')
    dist_matrix = dist_data['data'][0] / 1e6 # Shape (100, 100), divided by 1e6 as per data format

    lkh_data = np.load('/data/jinmohan/lkh/result_1k_euc/lkh_euc_TSP100_results.npy', allow_pickle=True).item()
    # lkh_data = np.load('/data/jinmohan/lkh/result_1k/sampled_symmetry_p0/lkh_ATSP100_results.npy', allow_pickle=True).item()
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
    
    num_random_walks = 50
    steps_per_walk = 10
    
    # Generate neighbor tours
    for _ in range(num_random_walks):
        curr_tour = opt_tour.copy()
        for step in range(steps_per_walk):
            curr_tour = two_opt_swap(curr_tour)
            samples.append(curr_tour)
            fitness.append(calculate_tour_length(curr_tour, dist_matrix))
            
    # Add some completely random tours to see the farther landscape
    for _ in range(100):
        rand_tour = list(range(100))
        random.shuffle(rand_tour)
        samples.append(np.array(rand_tour))
        fitness.append(calculate_tour_length(rand_tour, dist_matrix))

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
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_2d = mds.fit_transform(dist_mat)
    
    X = coords_2d[:, 0]
    Y = coords_2d[:, 1]
    Z = np.array(fitness)

    # 5. 2D Visualization
    print("Generating 2D Visualization...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Scatter plot
    scatter = ax.scatter(X, Y, c=Z, cmap='viridis', s=30, alpha=0.8, edgecolor='none')
    
    # Highlight the optimal solution
    ax.scatter(X[0], Y[0], color='red', s=150, marker='*', label=f'Optimal (Cost: {Z[0]:.2f})')
    
    ax.set_title("2D Fitness Landscape of TSP 100 (Sample 0)")
    ax.set_xlabel("Structural Dim 1 (MDS)")
    ax.set_ylabel("Structural Dim 2 (MDS)")
    
    fig.colorbar(scatter, ax=ax, label='Tour Length (Fitness)')
    ax.legend()
    
    out_path = '/data/jinmohan/RADAR/analysis/landscape_tsp100_2d.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {out_path}")

if __name__ == '__main__':
    main()