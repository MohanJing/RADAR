# 用于测试TSPLIB数据集的代码

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


import os
import sys
import torch
import numpy as np
import glob
import tsplib95
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=30, help='k iteration for sinkhorn')
args, _ = parser.parse_known_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
print(sys.path)

from utils.utils import create_logger, copy_all_src
from ATSPTester import ATSPTester as Tester

# Set random seeds
SEED = 1234

load_ckpt = 'radar_official_checkpoint'
ckpt_path = f'result/train/{load_ckpt}'

problem_cnt = 100

test_batch_size = {
    100: 1000,
    200: 400,
    500: 50,
    1000: 3   
}

aug_batch_size = {
    100: 20,
    200: 2
}

head_num = 8
embedding_dim = 256
qkv_dim = embedding_dim // head_num
##########################################################################################
# parameters

env_params = {
    'node_cnt':problem_cnt,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
    'pomo_size': problem_cnt  # same as node_cnt
}

model_params = {
    'embedding_dim': embedding_dim,
    'sqrt_embedding_dim': embedding_dim**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': qkv_dim,
    'sqrt_qkv_dim': qkv_dim**(1/2),
    'head_num': head_num,
    'init': 'svd',
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'softmax',
    'one_hot_seed_cnt': problem_cnt,  # must be >= node_cnt
    'k_iter': args.k,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        # 'path': 'result/radar_official_checkpoint',
        'path': ckpt_path,
        'epoch': 2100,
    },
    'npz_file': f'/data/jinmohan/NCOdata/atsp_1k/sampled_symmetry_p0/ATSP{problem_cnt}.npz', # 输入数据
    # 'npz_file': f'/data/jinmohan/RADAR/atsp/dataset/ATSP500.npz', # 输入数据
    'test_batch_size': test_batch_size[problem_cnt],
    'augmentation_enable': False,
    'aug_factor': 128, # 每个问题重复推理 128 次
    'aug_batch_size': 10, # 增强时的批大小
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'filepath': f'result/test/TSPLIB/{load_ckpt}_k={args.k}',
        'filename': 'log.txt'
    }
}

def load_tsplib_distance_matrix(filepath):
    problem = tsplib95.load(filepath)
    graph = problem.get_graph()
    dim = problem.dimension
    dist_matrix = np.zeros((dim, dim))
    
    nodes = list(graph.nodes)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if u != v and graph.has_edge(u, v):
                dist_matrix[i, j] = graph[u][v]['weight']
    return dist_matrix, nodes

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if DEBUG_MODE:
        tester_params['aug_factor'] = 10
        tester_params['test_batch_size'] = 10

    create_logger(**logger_params)
    
    # Minimal modification: Disable initial NPZ loading
    tester_params['npz_file'] = None
    tester = Tester(env_params, model_params, tester_params)
    
    tsp_dir = "/data/jinmohan/NCOdata/LIB/tsp"
    tsp_files = glob.glob(os.path.join(tsp_dir, "*.tsp"))
    
    # Load predefined optimal costs
    opt_costs_file = os.path.join(tsp_dir, "optimal_costs.json")
    predefined_costs = {}
    if os.path.exists(opt_costs_file):
        with open(opt_costs_file, 'r') as f:
            predefined_costs = json.load(f)
    
    gaps = []
    tester.logger.info(f"Start processing {len(tsp_files)} TSPLIB files from {tsp_dir}")
    for filepath in sorted(tsp_files):
        filename = os.path.basename(filepath)
        problem_name = filename.replace('.tsp', '')
        opt_tour_path = filepath.replace('.tsp', '.opt.tour')
        try:
            dist_matrix, nodes = load_tsplib_distance_matrix(filepath)
            route, cost = tester.test_single_instance(dist_matrix, problem_name=filename)
            
            opt_cost = None
            is_verified = False
            
            if os.path.exists(opt_tour_path):
                tour = tsplib95.load(opt_tour_path).tours[0]
                computed_opt_cost = 0
                for k in range(len(tour)):
                    u = tour[k]
                    v = tour[(k + 1) % len(tour)]
                    u_idx = nodes.index(u)
                    v_idx = nodes.index(v)
                    computed_opt_cost += dist_matrix[u_idx, v_idx]
                
                # Verify with predefined cost if available
                if problem_name in predefined_costs:
                    predefined = predefined_costs[problem_name]
                    if abs(computed_opt_cost - predefined) > 1e-3:
                        tester.logger.warning(f"[{filename}] Computed .opt.tour cost ({computed_opt_cost}) MISMATCHES predefined cost ({predefined})!")
                    else:
                        is_verified = True
                
                opt_cost = computed_opt_cost
            elif problem_name in predefined_costs:
                opt_cost = predefined_costs[problem_name]
                tester.logger.info(f"[{filename}] No .opt.tour found. Using predefined optimal cost.")
                
            if opt_cost is not None:
                gap = (cost - opt_cost) / opt_cost * 100
                gaps.append(gap)
                verification_tag = " [VERIFIED]" if is_verified else ""
                tester.logger.info(f"[{filename}] Opt Cost: {opt_cost:.4f}{verification_tag}, Gap: {gap:.2f}%")
            else:
                tester.logger.info(f"[{filename}] Opt Cost: N/A - No .opt.tour or predefined value found")
        except Exception as e:
            tester.logger.error(f"Failed to process {filename}: {e}")
            
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        tester.logger.info("*" * 40)
        tester.logger.info(f"Evaluation Finished! Average Gap on {len(gaps)} instances: {avg_gap:.2f}%")
        tester.logger.info("*" * 40)

if __name__ == "__main__":
    main()
