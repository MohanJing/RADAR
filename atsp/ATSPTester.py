import os
import sys
import torch
import numpy as np
from logging import getLogger

from ATSPEnv import ATSPEnv as Env

from ATSPModel import ATSPModel as Model
# from ACTmodel import ATSPModel as Model
# from RLmodel_layer import ATSPModel as Model

from utils.utils import get_result_folder, AverageMeter, TimeEstimator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.visualize_attention import visualize_attention_all_heads_raw_for_samples

class ATSPTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        if tester_params['use_cuda']:
            torch.cuda.set_device(tester_params['cuda_device_num'])
            self.device = torch.device('cuda', tester_params['cuda_device_num'])
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        checkpoint = torch.load(
            f"{tester_params['model_load']['path']}/checkpoint-{tester_params['model_load']['epoch']}.pt",
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()
        self.attn_visualized = True
        # self.visualize_sample_indices = set(self.tester_params.get('visualize_sample_indices', []))
        self.visualize_sample_indices = set([2316, 7414, 7600, 4959, 5021]) # 最差的5个样本索引
        self.visualized_sample_indices = set()

        # Load npz data
        if tester_params.get('npz_file'):
            npz_path = tester_params['npz_file']
            data = np.load(npz_path)['data']  # shape: (batch, node_cnt, node_cnt)
            data = data / (1000 * 1000)
            self.all_problems = torch.tensor(data, dtype=torch.float32)
        else:
            self.all_problems = None

    def run(self):
        self.time_estimator.reset()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        score_std_AM = AverageMeter()
        aug_score_std_AM = AverageMeter()

        episode = 0
        total = self.all_problems.size(0)
        batch_size = self.tester_params['test_batch_size']
        node_cnt = self.env_params['node_cnt']

        all_best_routes = np.zeros((total, node_cnt), dtype=np.int64)
        all_best_costs = np.zeros((total,), dtype=np.float32)

        while episode < total:
            bs = min(batch_size, total - episode)
            score, aug_score, score_std, aug_score_std, batch_best_routes, batch_best_costs = self._test_one_batch(episode, episode + bs)

            all_best_routes[episode:episode+bs] = batch_best_routes
            all_best_costs[episode:episode+bs] = batch_best_costs

            score_AM.update(score, bs)
            aug_score_AM.update(aug_score, bs)
            score_std_AM.update(score_std, bs)
            aug_score_std_AM.update(aug_score_std, bs)
            episode += bs

            elapsed, remain = self.time_estimator.get_est_string(episode, total)
            self.logger.info(
                f"Episode {episode}/{total}, Elapsed[{elapsed}], Remain[{remain}], "
                f"Score: {score:.3f}±{score_std:.3f}, Aug: {aug_score:.3f}±{aug_score_std:.3f}"
            )

        self.logger.info("*** Test Done ***")
        self.logger.info(f"NO-AUG SCORE: {score_AM.avg:.4f} ± {score_std_AM.avg:.4f}")
        self.logger.info(f"AUGMENTED SCORE: {aug_score_AM.avg:.4f} ± {aug_score_std_AM.avg:.4f}")

        # save_dir = os.path.join(os.path.dirname(__file__), f'result/answer/sampled_symmetry/p0/{node_cnt}')
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"{self.tester_params['model_load']['path'].split('/')[-1]}_k={self.model_params['k_iter']}.npz")
        # np.savez(save_path, tour=all_best_routes, cost=all_best_costs)
        # self.logger.info(f"Saved inference results to {save_path}")

    def _test_one_batch(self, idx_start, idx_end):
        problems = self.all_problems[idx_start:idx_end]  # shape: (batch, node, node)
        aug_factor = self.tester_params['aug_factor'] if self.tester_params['augmentation_enable'] else 1
        batch_size = 3

        all_rewards = []
        all_best_routes = []

        self.model.eval()
        with torch.no_grad():
            base = problems.size(0)

            num_batches = (aug_factor + batch_size - 1) // batch_size
            for i in range(num_batches):
                current_batch = min(batch_size, aug_factor - i * batch_size)

                problems_aug = problems.repeat(current_batch, 1, 1)  
                self.env.load_problems_manual(problems_aug)
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state)

                # 可视化注意力
                # target_sample_indices = [
                #     idx for idx in self.visualize_sample_indices
                #     if idx_start <= idx < idx_end and idx not in self.visualized_sample_indices
                # ]
                # if target_sample_indices:
                #     visualize_attention_all_heads_raw_for_samples(
                #         self.model,
                #         self.logger,
                #         self.result_folder,
                #         target_sample_indices,
                #         batch_start_idx=idx_start,
                #         layer_idx=4,
                #     )
                #     self.visualized_sample_indices.update(target_sample_indices)

                state, reward, done = self.env.pre_step()
                while not done:
                    selected, prob = self.model(state)
                    state, reward, done = self.env.step(selected)

                reward = reward.view(current_batch, base, self.env.pomo_size)
                selected_nodes = self.env.selected_node_list.view(
                    current_batch, base, self.env.pomo_size, self.env.node_cnt
                )

                best_pomo_idx = reward.argmax(dim=2)
                batch_idx = torch.arange(current_batch, device=reward.device)[:, None].expand(current_batch, base)
                problem_idx = torch.arange(base, device=reward.device)[None, :].expand(current_batch, base)
                best_routes = selected_nodes[batch_idx, problem_idx, best_pomo_idx]

                all_rewards.append(reward.max(dim=2)[0])
                all_best_routes.append(best_routes)

            all_rewards = torch.cat(all_rewards, dim=0)
            all_best_routes = torch.cat(all_best_routes, dim=0)

            best_aug_idx = all_rewards.argmax(dim=0)
            problem_idx = torch.arange(base, device=best_aug_idx.device)
            best_routes_overall = all_best_routes[best_aug_idx, problem_idx]
            best_costs = -all_rewards.max(dim=0)[0].float()

            no_aug = -all_rewards[0].float().mean()
            no_aug_std = all_rewards[0].float().std()

            aug = best_costs.mean()
            aug_std = best_costs.std()

            return (
                no_aug.item(),
                aug.item(),
                no_aug_std.item(),
                aug_std.item(),
                best_routes_overall.cpu().numpy(),
                best_costs.cpu().numpy(),
            )

    def test_single_instance(self, dist_matrix, problem_name=""):
        N = dist_matrix.shape[0]
        self.env_params['node_cnt'] = N
        self.env_params['pomo_size'] = min(N, 1000)
        self.env = Env(**self.env_params)
        
        # dist_matrix is a numpy array of shape (N, N). Add batch dimension
        data = np.expand_dims(dist_matrix, axis=0)
        self.all_problems = torch.tensor(data, dtype=torch.float32)
        
        score, aug_score, score_std, aug_score_std, batch_best_routes, batch_best_costs = self._test_one_batch(0, 1)
        
        self.logger.info(f"[{problem_name}] Nodes: {N}, POMO: {self.env_params['pomo_size']}, Cost: {batch_best_costs[0]:.4f}")
        return batch_best_routes[0], batch_best_costs[0]