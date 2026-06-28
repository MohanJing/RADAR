import torch
from logging import getLogger
from ATSPEnv import ATSPEnv as Env, Reset_State
import wandb
from utils.utils import *
from RLmodel import ATSPModel

class RLTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.env = Env(**self.env_params)
        
        self.model = ATSPModel(**self.model_params)
        
        # 【关键修改】建立两个独立的优化器
        self.supernet_optim = torch.optim.Adam(self.model.get_supernet_params(), lr=self.optimizer_params['supernet']['lr'])
        self.controller_optim = torch.optim.Adam(self.model.get_controller_params(), lr=self.optimizer_params['controller']['lr']) # Controller 可以用稍大的 LR

        # --- 单独加载 Supernet 预训练权重 ---
        supernet_load = trainer_params.get('supernet_load', {})
        if supernet_load.get('enable', False):
            ckpt = torch.load(supernet_load['path'], map_location=device)
            # 只取 encoder/decoder 相关参数，排除 controller
            supernet_state = {k: v for k, v in ckpt['model_state_dict'].items()
                              if 'controller' not in k}
            missing, unexpected = self.model.load_state_dict(supernet_state, strict=False)
            self.logger.info(f'Supernet loaded from {supernet_load["path"]}')
            self.logger.info(f'  Missing keys (controller, expected): {missing}')
            self.logger.info(f'  Unexpected keys: {unexpected}')

        # --- 冻结 Supernet，只训练 Controller ---
        self.freeze_supernet = trainer_params.get('freeze_supernet', False)
        if self.freeze_supernet:
            for param in self.model.get_supernet_params():
                param.requires_grad = False
            self.logger.info('Supernet frozen — only Controller will be trained')

        # Restore (full checkpoint, including both optimizers)
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.supernet_optim.load_state_dict(checkpoint['supernet_optimizer_state_dict'])
            self.controller_optim.load_state_dict(checkpoint['controller_optimizer_state_dict'])
            self.logger.info('Full checkpoint loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

        # EMA Baseline 初始化
        self.ema_baseline = None
        
        # 两阶段交替训练配置：从 trainer_params 读取，支持用户自定义
        self.supernet_epochs = self.trainer_params.get('supernet_epochs', 5)
        self.controller_epochs = self.trainer_params.get('controller_epochs', 1)
        self.beta_entropy = self.trainer_params.get('beta_entropy', 0.05)
        self.ponder_lambda = self.trainer_params.get('ponder_lambda', 0.005)
        
        wandb_config = {
            'env_params': self.env_params,
            'model_params': self.model_params,
            'optimizer_params': self.optimizer_params,
            'trainer_params': self.trainer_params
        }
        
        wandb.init(
            project="RADAR_two_stage",          # 在 wandb 上的项目名称
            name=f"RL_two_stage", # 运行的名称
            config=wandb_config,          # 记录超参数
        )

    def run(self):
        for epoch in range(1, self.trainer_params['epochs']+1):

            # 确定当前 Epoch 处于哪个训练阶段
            if self.freeze_supernet:
                # Supernet 冻结时全程训练 Controller
                phase = "controller"
            else:
                # 两阶段交替：supernet_epochs 个 epoch 训练 Supernet，
                #              controller_epochs 个 epoch 训练 Controller
                cycle = self.supernet_epochs + self.controller_epochs
                rem = epoch % cycle
                if rem == 0:
                    phase = "controller"
                elif rem <= self.supernet_epochs:
                    phase = "supernet"
                else:
                    phase = "controller"
                
            self.logger.info(f'--- Epoch {epoch} | Phase: {phase.upper()} ---')

            # Train
            train_score, train_loss, avg_K = self._train_one_epoch(epoch, phase)

            # Wandb 记录
            wandb.log({
                "epoch": epoch,
                "phase": 1 if phase == "supernet" else 2,
                "train_score": train_score,
                f"{phase}_loss": train_loss,
                "avg_step_K": avg_K,
                "ema_baseline": self.ema_baseline if self.ema_baseline else 0.0
            }, step=epoch)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'supernet_optimizer_state_dict': self.supernet_optim.state_dict(),
                    'controller_optimizer_state_dict': self.controller_optim.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

                # 结束 WANDB 进程
                wandb.finish()

    def _train_one_epoch(self, epoch, phase):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        avg_K_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, avg_K = self._train_one_batch(batch_size, phase)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            avg_K_AM.update(avg_K, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}, Avg_K: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg, avg_K_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}, Avg_K: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg, avg_K_AM.avg))

        return score_AM.avg, loss_AM.avg, avg_K_AM.avg

    def _train_one_batch(self, batch_size, phase):
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()

        state, reward, done = self.env.pre_step()

        if phase == "supernet":
            # 必须调用 pre_forward 触发随机 K 采样
            self.model.pre_forward(reset_state, phase=phase)
            prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0), device=self.model.decoder.Wk.weight.device)
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            advantage_supernet = reward - reward.float().mean(dim=1, keepdims=True) # shape: (batch, pomo)
            log_prob = prob_list.log().sum(dim=2)
            loss_supernet = (-advantage_supernet * log_prob).mean()

            max_pomo_reward, _ = reward.max(dim=1) # shape: (batch,) 选择pomo中最好的那个
            score_mean = -max_pomo_reward.float().mean()

            # 只更新 Encoder-Decoder
            self.supernet_optim.zero_grad()
            loss_supernet.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_supernet_params(), 1.0)
            self.supernet_optim.step()
            
            return score_mean.item(), loss_supernet.item(), self.model.K_samples.abs().float().mean().item()

        elif phase == "controller":
            # 阶段二：利用 Shared Baseline 更新 Controller
            # 【修复1】：必须先执行 controller 的前向，使其根据图特征采样出 K_samples 和对应的 log_prob
            self.model.pre_forward(reset_state, phase=phase)

            # 安全检查：确保 Controller 已经正确采样了 K
            if self.model.dist is None :
                raise RuntimeError(
                    "Make sure pre_forward was called with phase='controller'."
                )

            # 必须在进入 Baseline 循环前，将 Controller 输出的计算图和数据缓存！
            # 避免被 _compute_shared_baseline 中的 forced_K 前向操作污染。
            dist = self.model.dist
            probs_all_K = dist.probs          # shape: (batch, num_K)
            log_probs_all_K = dist.logits     # shape: (batch, num_K)

            # 【获取正确的 Entropy】
            entropy = dist.entropy()          # shape: (batch,)
            
            # 【缓存真实的采样 K 用于最终的日志记录】
            sampled_K = self.model.K_samples  # shape: (batch,)

            # --- Shared Baseline 计算 ---
            # 对每个可能的 K 值评估 reward，得到 (batch, 2*max_steps) 的 per-K reward，
            # 然后求平均作为实例相关的 baseline（类似 POMO 的 shared baseline 思想）
            all_rewards, shared_baseline = self._compute_shared_baseline(batch_size, reset_state)

            # 【修复2：构建全空间的 Ponder 惩罚】
            max_steps = self.model.max_steps
            all_K = torch.cat([
                torch.arange(-max_steps, 0),
                torch.arange(1, max_steps + 1)
            ]).to(device=all_rewards.device)
            
            all_K_abs = all_K.abs().float() # shape: (num_K,)
            
            # Advantage = 真实 reward - 基线 - 对每一个 K 分别的惩罚
            # all_K_abs.unsqueeze(0) 会广播为 (batch, num_K)
            adv_all_K = (all_rewards - shared_baseline) - self.ponder_lambda * all_K_abs.unsqueeze(0)

            # 4. 计算解析策略梯度 Loss
            loss_controller = - (adv_all_K.detach() * probs_all_K.detach() * log_probs_all_K).sum(dim=1).mean() \
                              - self.beta_entropy * entropy.mean()

            self.controller_optim.zero_grad()
            loss_controller.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_controller_params(), 1.0)
            self.controller_optim.step()

            max_K_reward, _ = all_rewards.max(dim=1) # shape: (batch,) 选择所有K中最好的那个
            score_mean = -max_K_reward.float().mean()

            # 使用一开始缓存的 sampled_K 来反映 Controller 真实的采样偏好
            return score_mean.item(), loss_controller.item(), sampled_K.abs().float().mean().item()

    def _compute_shared_baseline(self, batch_size, reset_state):
        """
        POMO-style shared baseline for Controller RL.

        对每个 K ∈ [-max_steps, -1] ∪ [1, max_steps] 评估 reward，
        取 max over pomo → average over K，得到实例相关的 baseline (batch,)。

        """
        max_steps = self.model.max_steps
        saved_problems = reset_state.problems.clone()
        device = saved_problems.device

        # 生成所有 K 值列表：[-max_steps, ..., -1, 1, ..., max_steps]
        all_K = torch.cat([
            torch.arange(-max_steps, 0),
            torch.arange(1, max_steps + 1)
        ])  # (2*max_steps,)
        num_K = len(all_K)

        all_rewards = torch.zeros(batch_size, num_K, device=device)  # (batch, 2*max_steps)

        # 【核心参数】：并行分组大小。
        # 取值范围 1 到 num_K。
        # 如果设为 1，等价于纯串行；如果设为 num_K，则是全量并行。
        # 建议先设定为 8 观察显存（3090 24G 预计能承受 8-16 的并行度）。
        chunk_size = 16  

        with torch.no_grad():
            for i in range(0, num_K, chunk_size):
                # 提取当前并行块对应的 K 值
                chunk_K = all_K[i : i + chunk_size]
                current_chunk = len(chunk_K)

                # 1. 复制图特征：将 problems 沿着 batch 维度复制 current_chunk 份
                # 形状变为: (current_chunk * batch_size, node, node)
                expanded_problems = saved_problems.repeat(current_chunk, 1, 1)

                # 2. 对齐 K 值掩码
                # chunk_K 形状: (current_chunk,) 
                # 变换后 forced_K 形状: (current_chunk * batch_size,)
                # 结构为: [K0, K0..., K1, K1..., K2, K2...] (每个 K 连续 batch_size 次)
                forced_K = chunk_K.unsqueeze(1).expand(current_chunk, batch_size).reshape(-1)

                # 3. 前向计算（利用扩充后的并行 Batch）
                eval_rs = Reset_State(problems=expanded_problems)
                self.model.pre_forward(eval_rs, forced_K_samples=forced_K)

                self.env.load_problems_manual(expanded_problems)
                state, reward, done = self.env.reset()
                state, reward, done = self.env.pre_step()
                
                while not done:
                    selected, prob = self.model(state)
                    state, reward, done = self.env.step(selected)

                # 4. 提取当前块的结果
                max_r, _ = reward.max(dim=1)  # shape: (current_chunk * batch_size,)
                
                # 5. 结果重排与赋值
                # 先转化为 (current_chunk, batch_size)，然后转置回 (batch_size, current_chunk)
                chunk_rewards = max_r.view(current_chunk, batch_size).transpose(0, 1)
                
                # 写入最终的 Reward 矩阵中
                all_rewards[:, i : i + current_chunk] = chunk_rewards

        # 对每个 K，取 pomo 中的最大值（已在上面的 max 中完成）
        # all_rewards: (batch, 2*max_steps)
        # Shared baseline = mean over K
        shared_baseline = all_rewards.mean(dim=1, keepdim=True)  # (batch, 1)

        return all_rewards, shared_baseline