
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# ==========================================
# 设置确定性环境变量（必须在导入 torch 之前）
# ==========================================
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'


##########################################################################################
problems_size = 100
head_num = 8
embedding_dim = 256
name = 'rl_two_stage_6feat'
qkv_dim = embedding_dim // head_num 

##########################################################################################
# Path Config

import sys
import random
import numpy as np
import torch
 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
print(sys.path)

##########################################################################################
# import

import logging

from utils.utils import create_logger, copy_all_src
from RLTrainer import RLTrainer as Trainer
# from ACTTrainer import ATSPTrainer as Trainer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 PyTorch 的确定性算法（warn_only=True 允许某些操作仍可运行）
    torch.use_deterministic_algorithms(True, warn_only=True)
##########################################################################################
# parameters


env_params = {
    'node_cnt': problems_size,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
    'pomo_size': problems_size,  # same as node_cnt
    'mix_tsp': False
}

model_params = {
    'embedding_dim': embedding_dim,
    'sqrt_embedding_dim': embedding_dim**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': qkv_dim,
    'sqrt_qkv_dim': qkv_dim**(1/2),
    'head_num': head_num,
    'init': 'svd',
    'att_type': 'normal',
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (3/3)**(1/2),
    'ms_layer2_init': (3/16)**(1/2),
    'max_steps': 40,           # Controller 可预测的最大 Sinkhorn 迭代步数 K ∈ [1, max_steps]
    'eval_type': 'softma',
    'one_hot_seed_cnt': problems_size,  # must be >= node_cnt
}

optimizer_params = {
    # RLTrainer 内部创建两个独立优化器，此处记录实际使用的参数
    'supernet': {
        'lr': 4*1e-4,           # Encoder-Decoder 学习率
        'weight_decay': 1e-6
    },
    'controller': {
        'lr': 1e-4,           # StepController 学习率（更大以加速策略收敛）
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [2001, 2101],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 2100,
    'train_episodes': 10*1000,
    'train_batch_size': 64,
    # 两阶段交替训练控制
    'supernet_epochs': 5,        # 每周期 Supernet 训练轮数
    'controller_epochs': 1,      # 每周期 Controller 训练轮数
    'beta_entropy': 0.01,        # Controller 熵正则化系数（鼓励探索）
    'ponder_lambda': 0.000,      # 迭代步数惩罚系数（越大越鼓励用小 K）
    # ε-greedy 衰减：supernet 阶段以 ε 概率均匀探索，(1-ε) 概率使用 Controller 利用
    'eps_start': 1.0,            # 初始 ε（早期纯探索）
    'eps_end': 0.05,             # 最终 ε（后期偏利用，保留 5% 探索维持鲁棒性）
    'eps_decay_epochs': 2100,    # ε 线性衰减的 epoch 数（设为总 epoch 数即全程衰减）
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 200,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading full checkpoint (supernet + controller)
        'path': '',       # directory path of pre-trained model and log files saved.
        'epoch': 800,     # epoch version of pre-trained model to load.
    },
    # 单独加载 Supernet 预训练权重（不含 Controller），用于两阶段训练
    'supernet_load': {
        'enable': False,  # 设为 True 加载预训练 supernet
        'path': '/data/jinmohan/RADAR/atsp/result/train/radar_official_checkpoint/checkpoint-2100.pt',       # .pt 文件完整路径
    },
    # 冻结 Supernet，只训练 Controller
    'freeze_supernet': False,
}

logger_params = {
    'log_file': {
        'desc': name,
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    # 设置随机种子
    SEED = 1234
    set_seed(SEED)
    print(f'Random seed set to: {SEED}')

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    # copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():

    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2
    trainer_params['validate_episodes'] = 4
    trainer_params['validate_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
