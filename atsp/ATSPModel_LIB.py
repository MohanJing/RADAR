from ACTsinkhorn import ACTSinkhorn
import torch
import torch.nn as nn
import torch.nn.functional as F


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 3, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 3, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

        self.k_iter = model_params.get('k_iter', 20)
        # 初始化 ACT Sinkhorn，设定一个合理的最大允许单次操作上限，如 20 次操作 (=10个完整行列迭代)
        self.act_sinkhorn = ACTSinkhorn(max_steps=20, eps=1e-3)

    def forward(self, q, k, v, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)
        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score, cost_mat_score.transpose(2,3)), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 3)

        two_scores_transposed = two_scores.transpose(1,2)
        # shape: (batch, row_cnt, head_num, col_cnt, 3)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1,2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # mixed_scores = dot_product / sqrt_qkv_dim  # 直接使用原始score，不用混合得分
        # shape: (batch, head_num, row_cnt, col_cnt)

        # ========= 保存归一化前的原始打分 (Raw Scores) =========
        # self.last_raw_scores = mixed_scores.detach().cpu()
        
        # 顺便提前算一下 dim=2 和 dim=3 的 Softmax 单纯拿去画图用
        # self.last_softmax_dim2 = F.softmax(mixed_scores, dim=2).detach().cpu()
        # self.last_softmax_dim3 = F.softmax(mixed_scores, dim=3).detach().cpu()
        # =======================================================

        # weights = nn.Softmax(dim=3)(mixed_scores)
        weights = self.act_sinkhorn(mixed_scores)
        # weights = sinkhorn_normalization_k(mixed_scores, k_iter=self.k_iter) 
        # weights = sinkhorn_normalization(mixed_scores)

        # ========= 保存归一化之后的注意力权重 (Attention Weights) =========
        # self.last_attention_weights = weights.detach().cpu()
        # =================================================================

        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)
        
        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat
    

def sinkhorn_normalization(scores, n_iter=10):
    scores = scores - scores.max(dim=-1, keepdim=True)[0]
    for _ in range(n_iter):
        scores = scores - scores.logsumexp(dim=-1, keepdim=True)
        scores = scores - scores.logsumexp(dim=-2, keepdim=True)
    return scores.exp()

def sinkhorn_normalization_k(scores, k_iter=20):
    # 过滤无效输入，k=0 时直接返回非负权重（假设需求如此）
    if k_iter == 0:
        return torch.ones_like(scores) / scores.shape[-1]
        
    num_steps = abs(k_iter)
    start_dim = -2 if k_iter < 0 else -1
    
    # 【修复点】：初始稳定性处理必须沿着起始维度 start_dim 进行
    scores = scores - scores.max(dim=start_dim, keepdim=True)[0]
    
    for i in range(num_steps):
        current_dim = start_dim if i % 2 == 0 else (-3 - start_dim) 
        # logsumexp 自带数值稳定性，连续调用不会导致溢出
        scores = scores - scores.logsumexp(dim=current_dim, keepdim=True)
        
    return scores.exp()