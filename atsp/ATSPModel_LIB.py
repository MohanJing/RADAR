
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

        # =========================================================
        # [新增] 边特征到 Value 空间的投影权重
        # 为了稳定训练，使用较小的方差初始化，让网络逐渐学习边特征的融入
        # =========================================================
        qkv_dim = self.model_params['qkv_dim']
        
        # 映射前向距离 cost_mat_{ij} (i 到 j 的成本)
        self.W_v_edge_fwd = nn.Parameter(torch.randn(head_num, qkv_dim) * 0.02)
        # 映射反向距离 cost_mat_{ji} (j 到 i 的成本)
        self.W_v_edge_bwd = nn.Parameter(torch.randn(head_num, qkv_dim) * 0.02)

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

        # weights = nn.Softmax(dim=3)(mixed_scores)
        weights = sinkhorn_normalization(mixed_scores)

        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)
        
        # =========================================================
        # 新增
        # 1. 计算前向和反向的距离矩阵，并强制转换为连续内存 (彻底杜绝寻址错误)
        # cost_fwd shape: (batch, 1, row_cnt, col_cnt)
        cost_fwd = cost_mat.unsqueeze(1) 
        # cost_bwd shape: (batch, 1, col_cnt, row_cnt) -> 注意这里的 .contiguous() 是救命的
        cost_bwd = cost_mat.transpose(1, 2).contiguous().unsqueeze(1)

        # 2. 先加权求和计算标量聚合 (利用广播机制)
        # S_fwd/S_bwd shape: (batch, head_num, row_cnt)
        S_fwd = torch.sum(weights * cost_fwd, dim=3)
        S_bwd = torch.sum(weights * cost_bwd, dim=3)

        # 3. 最后乘以投影权重映射到高维空间
        # self.W_v_edge_fwd shape: (head_num, qkv_dim)
        # 利用 unsqueeze 扩展维度进行广播乘法: [B, H, R, 1] * [1, H, 1, D] -> [B, H, R, D]
        out_edge_fwd = S_fwd.unsqueeze(3) * self.W_v_edge_fwd.unsqueeze(0).unsqueeze(2)
        out_edge_bwd = S_bwd.unsqueeze(3) * self.W_v_edge_bwd.unsqueeze(0).unsqueeze(2)

        # 4. 联合融合
        out = out + out_edge_fwd + out_edge_bwd
        # shape: (batch, head_num, row_cnt, qkv_dim)

        # =========================================================


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