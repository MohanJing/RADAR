import torch
from logging import getLogger
from ATSPEnv import ATSPEnv as Env
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
        self.supernet_optim = torch.optim.Adam(self.model.get_supernet_params(), lr=1e-4)
        self.controller_optim = torch.optim.Adam(self.model.get_controller_params(), lr=1e-3) # Controller 可以用稍大的 LR
        
        # Restore
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
            self.logger.info('Saved Model Loaded !!')

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
            name=f"RL_train", # 运行的名称
            config=wandb_config,          # 记录超参数
        )

    def run(self):
        for epoch in range(1, self.trainer_params['epochs']+1):
            
            # 确定当前 Epoch 处于哪个训练阶段
            # 两阶段交替：supernet_epochs 个 epoch 训练 Supernet，
            #              controller_epochs 个 epoch 训练 Controller
            cycle = self.supernet_epochs + self.controller_epochs
            rem = epoch % cycle
            if rem == 0:
                phase = "controller"  # cycle 的最后一个 epoch（如 epoch 6, 12, ...）
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
        
        # 传递 phase 标识，控制采样逻辑
        self.model.pre_forward(reset_state, phase=phase)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0), device=self.model.decoder.Wk.weight.device)
        
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # 无论哪个阶段，都需要计算 POMO 的 Advantage
        advantage_supernet = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)
        loss_supernet = (-advantage_supernet * log_prob).mean()

        max_pomo_reward, _ = reward.max(dim=1) # shape: (batch,)
        score_mean = -max_pomo_reward.float().mean()

        # 记录 EMA Baseline，仅用作 Controller 的基线评估
        if self.ema_baseline is None:
            self.ema_baseline = max_pomo_reward.mean().item()
        else:
            self.ema_baseline = 0.95 * self.ema_baseline + 0.05 * max_pomo_reward.mean().item()

        if phase == "supernet":
            # 阶段一：只更新 Encoder-Decoder
            self.supernet_optim.zero_grad()
            loss_supernet.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_supernet_params(), 1.0)
            self.supernet_optim.step()
            
            return score_mean.item(), loss_supernet.item(), self.model.K_samples.float().mean().item()

        elif phase == "controller":
            # 阶段二：利用 Rollout Baseline 更新 Controller

            # 安全检查：确保 Controller 已经正确采样了 K
            if self.model.K_log_prob is None or self.model.K_entropy is None:
                raise RuntimeError(
                    "Controller phase requires K_log_prob and K_entropy to be set. "
                    "Make sure pre_forward was called with phase='controller'."
                )

            # Controller Advantage = 实际最好的路径 Reward - EMA 基准 Reward - Ponder 惩罚
            adv_controller = (max_pomo_reward - self.ema_baseline) - self.ponder_lambda * self.model.K_samples.float()

            # Controller RL Loss = - Advantage * log_prob - beta * entropy
            loss_controller = -(adv_controller.detach() * self.model.K_log_prob).mean() \
                              - self.beta_entropy * self.model.K_entropy.mean()
                              
            self.controller_optim.zero_grad()
            loss_controller.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_controller_params(), 1.0)
            self.controller_optim.step()

            return score_mean.item(), loss_controller.item(), self.model.K_samples.float().mean().item()