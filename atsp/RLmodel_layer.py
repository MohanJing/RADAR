import torch
import torch.nn as nn
import torch.nn.functional as F

from ATSPModel_LIB import *
from RLsinkhorn import StepController


def _idx_to_K(idx, max_steps):
    """将类别索引 [0, 2*max_steps-1] 映射为 K ∈ [-max_steps, -1] ∪ [1, max_steps]

    Args:
        idx: (batch,) LongTensor, category indices in [0, 2*max_steps-1]
        max_steps: int, maximum absolute K value

    Returns:
        K: (batch,) LongTensor, signed K values
    """
    return torch.where(
        idx < max_steps,
        idx - max_steps,       # 负: [-max_steps, -1]
        idx - max_steps + 1    # 正: [1, max_steps]
    )


class ATSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.max_steps = self.model_params.get('max_steps', 20)

        self.encoder = ATSP_Encoder(**model_params)
        self.decoder = ATSP_Decoder(**model_params)
        # extra_dim=5: 本层原始得分统计量(4) + 层数特征(1)
        # [旧] 原始特征: layer_feat(embedding_dim) + raw_stats(4) + layer_idx(1) = emb+5
        # self.controller = StepController(self.embedding_dim, self.max_steps, extra_dim=5)
        # [新] 特征: frob_err(1) + raw_stats(4) + layer_idx(1) = 6
        self.controller = StepController(0, self.max_steps, extra_dim=6)

        self.k = 10
        self.projection = nn.Linear(2 * self.k, self.embedding_dim)
        self.test = self.model_params.get('test', False)

    def get_supernet_params(self):
        # 返回除了 Controller 以外的所有参数
        return [p for n, p in self.named_parameters() if 'controller' not in n]

    def get_controller_params(self):
        return self.controller.parameters()

    def pre_forward(self, reset_state, phase="supernet", forced_K_samples=None):
        # Encoder阶段
        """
        两阶段训练的前向预处理。

        Phase "supernet" (阶段一):
          - 均匀随机采样 K ∈ [-max_steps, -1] ∪ [1, max_steps]
          - 强制打破表征耦合，让 Encoder-Decoder 学会在各种迭代步数下都能工作
          - K 不参与梯度计算 (detached)

        Phase "controller" (阶段二):
          - 利用 StepController 根据图特征预测最优 K
          - 通过 RL 训练 Controller：奖励 = 解质量 - ponder_cost
          - K 的 log_prob 和 entropy 被保存用于 RL loss

        forced_K_samples: 可选 (batch,) LongTensor，强制使用指定的 K 值，
          用于 shared baseline 计算时枚举所有可能的 K。
        """
        problems = reset_state.problems
        batch_size = problems.size(0)

        # 原始数据的相对 Frobenius 误差: ||D - D^T||_F / ||D||_F（归一化之前计算）
        raw_fro_norm = torch.sum(problems ** 2, dim=(1, 2))       # (batch,)
        raw_asym_norm = torch.sum((problems - problems.transpose(1, 2)) ** 2, dim=(1, 2))  # (batch,)
        frob_err = torch.sqrt(raw_asym_norm / (raw_fro_norm + 1e-8)).unsqueeze(1)  # (batch, 1)

        # ... (保留原有的 SVD 或 TopK 初始化逻辑) ...
        init_method = self.model_params.get('init', 'svd')
        
        mean_val = problems.mean(dim=(1, 2), keepdim=True)
        std_val = problems.std(dim=(1, 2), keepdim=True)
        problems = (problems - mean_val) / (std_val + 1e-9)
        
        if init_method == 'svd':
            U, S, V = torch.svd_lowrank(problems, q=self.k)
            sqrt_S = torch.sqrt(S)  # (batch, k)

            Q = U * sqrt_S.unsqueeze(1)  # (batch, n, k)
            K = V * sqrt_S.unsqueeze(1)  # (batch, n, k)

            X = torch.cat([Q, K], dim=-1)  # (batch, n, 2*k)

            final_embedding = self.projection(X)
        elif init_method == 'topk':
            n = problems.size(1)
            k_eff = min(self.k, max(n - 1, 0))

            if n > 1:
                diag_mask = torch.eye(n, device=problems.device, dtype=torch.bool).unsqueeze(0)
                masked_problems = problems.masked_fill(diag_mask, float('inf'))
            else:
                masked_problems = problems

            if k_eff > 0:
                out_topk = torch.topk(masked_problems, k=k_eff, dim=2, largest=False).values
                in_topk = torch.topk(
                    masked_problems.transpose(1, 2), k=k_eff, dim=2, largest=False
                ).values
            else:
                out_topk = torch.empty(batch_size, n, 0, device=problems.device, dtype=problems.dtype)
                in_topk = torch.empty(batch_size, n, 0, device=problems.device, dtype=problems.dtype)

            if k_eff < self.k:
                pad = torch.zeros(
                    batch_size, n, self.k - k_eff, device=problems.device, dtype=problems.dtype
                )
                out_topk = torch.cat([out_topk, pad], dim=-1)
                in_topk = torch.cat([in_topk, pad], dim=-1)

            X = torch.cat([out_topk, in_topk], dim=-1)  # (batch, n, 2*k)

            final_embedding = self.projection(X)
        
        # 将归一化后的 cost_mat 传递给 Encoder
        # Encoder 内部按层独立决定 K：
        #   supernet:   ε-greedy（ε 均匀探索 + (1-ε) Controller 利用）
        #   controller: 每层由 Controller 根据当前层输入隐向量特征预测不同的 K
        #   forced_K_samples: 所有层使用相同的强制 K 值
        eps = getattr(self, 'eps', 1.0)  # trainer 在每 epoch 开始时设置
        if self.test:
            eps = 0.0  # 测试阶段完全利用 Controller 预测的 K

        self.encoded_node = self.encoder(final_embedding, problems,
                                          controller=self.controller,
                                          phase=phase,
                                          forced_K_samples=forced_K_samples,
                                          eps=eps, test=self.test,
                                          frob_err=frob_err)

        # 从 Encoder 获取逐层的 dist 和 K_samples
        self.layer_dists = self.encoder.layer_dists           # list of Categorical | None
        self.layer_K_samples = self.encoder.layer_K_samples   # list of (batch,) LongTensor
        self.layer_K_samples_idx = self.encoder.layer_K_samples_idx

        # 保持向后兼容的接口
        self.dist = self.layer_dists
        self.K_samples = torch.stack(self.layer_K_samples, dim=0)  # (num_layers, batch)
        self.decoder.set_kv(self.encoded_node)

    def forward(self, state):
        # Decoder阶段
        
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_row = _get_encoding(self.encoded_node, selected)
            self.decoder.set_q1(encoded_first_row)

        else:
            encoded_current_node = _get_encoding(self.encoded_node, state.current_node)
            # shape: (batch, pomo, embedding)
            all_job_probs = self.decoder(encoded_current_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, job)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True: 
                    with torch.no_grad():
                        selected = all_job_probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)

                    prob = all_job_probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break
            else:
                selected = all_job_probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = torch.zeros_like(selected, dtype=torch.float32) 

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################
class ATSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.max_steps = model_params.get('max_steps', 20)
        self.layers = nn.ModuleList([EncodingBlock(**model_params) for _ in range(encoder_layer_num)])
        self.num_layers = encoder_layer_num
        self.ponder_cost = 0.0  # 累计所有层的平均 Sinkhorn 迭代步数
        self.layer_dists = []       # 逐层 Categorical 分布（或 None）
        self.layer_K_samples = []   # 逐层 K 值
        self.layer_K_samples_idx = []

    def forward(self, node_emb, cost_mat, controller=None, phase="supernet",
                forced_K_samples=None, eps=1.0, test=False, frob_err=None):
        """Encoder 前向传播，按层独立决定 K 值。

        每层采用两阶段前向：
          阶段一：计算本层原始混合得分矩阵并提取统计量（残差+方差）
          阶段二：Controller 根据 [隐向量均值 | 原始得分统计量] 预测该层 K，
                  然后执行 Sinkhorn 完成该层计算

        Phase "supernet" — ε-greedy 混合采样:
          - 以概率 ε：每层独立均匀随机 K（探索，维持鲁棒性）
          - 以概率 1-ε：Controller 预测 K（利用，针对性微调）
          - ε 随训练 epoch 从 1.0 线性衰减到 eps_end

        Args:
            node_emb: (batch, node_cnt, embedding_dim) 初始节点嵌入
            cost_mat: (batch, node_cnt, node_cnt) 代价矩阵
            controller: StepController 模块（controller 阶段使用）
            phase: "supernet" | "controller"
            forced_K_samples: 可选 (batch,) LongTensor，强制所有层使用相同 K
            eps: float, ε-greedy 探索概率（仅 supernet 阶段有效）

        Returns:
            node_emb: (batch, node_cnt, embedding_dim) 编码后的节点嵌入
        """
        batch_size = node_emb.size(0)
        num_categories = 2 * self.max_steps
        total_ponder_cost = 0.0
        layer_dists = []
        layer_K_samples = []
        layer_K_samples_idx = []

        for layer_idx, layer in enumerate(self.layers):
            # ===== 阶段一：计算本层原始得分矩阵的统计量（Sinkhorn 之前） =====
            raw_stats = layer.prepare_scores(node_emb, cost_mat)  # (batch, 4)

            # ===== 阶段二：决定本层的 K 值 =====
            # [旧] 提取当前层输入隐向量的特征：所有节点隐向量的均值
            # layer_feat = node_emb.mean(dim=1)  # (batch, embedding_dim)
            # [新] 使用原始数据的相对 Frobenius 误差替代 layer_feat

            if forced_K_samples is not None:
                # Shared baseline 模式：所有层使用外部强制指定的 K 值
                dist = None
                K_samples = forced_K_samples.to(device=node_emb.device)
                K_samples_idx = None
            elif phase == "supernet":
                # ε-greedy：每层独立，每个实例 ε 概率均匀探索，(1-ε) 概率 Controller 利用
                # 始终计算 Controller 分布（用于 1-ε 部分）
                layer_idx_feat = torch.full((batch_size, 1),
                                            (layer_idx + 1) / self.num_layers,
                                            device=node_emb.device)
                # [旧] combined_feat = torch.cat([layer_feat, raw_stats, layer_idx_feat], dim=-1)
                combined_feat = torch.cat([frob_err, raw_stats, layer_idx_feat], dim=-1)
                with torch.no_grad():
                    dist = controller(combined_feat)  # Categorical over 2*max_steps
                    if test:
                        controller_idx = dist.probs.argmax(dim=1)  # 测试阶段选择概率最高的 K
                        # controller_idx = dist.sample()
                    else:
                        controller_idx = dist.sample()
                    controller_K = _idx_to_K(controller_idx, self.max_steps)

                # 均匀探索样本
                uniform_idx = torch.randint(0, num_categories, (batch_size,), device=node_emb.device)
                uniform_K = _idx_to_K(uniform_idx, self.max_steps)

                # 每个实例独立决定：ε → uniform, 1-ε → controller
                use_uniform = torch.rand(batch_size, device=node_emb.device) < eps
                K_samples_idx = torch.where(use_uniform, uniform_idx, controller_idx)
                K_samples = torch.where(use_uniform, uniform_K, controller_K)
            else:
                # 阶段二：Controller 根据 [隐向量均值 | 本层原始得分统计量 | 层数] 预测该层 K
                # 层数特征: 从 1 开始归一化到 (0, 1]
                layer_idx_feat = torch.full((batch_size, 1),
                                            (layer_idx + 1) / self.num_layers,
                                            device=node_emb.device)
                # [旧] combined_feat = torch.cat([layer_feat, raw_stats, layer_idx_feat], dim=-1)
                combined_feat = torch.cat([frob_err, raw_stats, layer_idx_feat], dim=-1)
                # shape: (batch, 6)
                dist = controller(combined_feat)  # Categorical over 2*max_steps categories
                K_samples_idx = dist.sample()
                K_samples = _idx_to_K(K_samples_idx, self.max_steps)

            layer_dists.append(dist)
            layer_K_samples.append(K_samples)
            layer_K_samples_idx.append(K_samples_idx)

            # ===== 阶段三：用预测的 K 执行 Sinkhorn 并完成该层计算 =====
            node_emb = layer.execute_with_K(K_samples)
            total_ponder_cost += layer.ponder_cost  # 累积每层的计算时间成本

        self.ponder_cost = total_ponder_cost
        self.layer_K_samples_idx = layer_K_samples_idx
        self.layer_dists = layer_dists
        self.layer_K_samples = layer_K_samples

        return node_emb



class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=True)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=True)
        # 以下遵循MatNet的实现，Wq, Wk, Wv 都使用 bias=False
        # self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.mixed_score_MHA = MixedScore_MultiHeadAttention_dynamic(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

        self.g_layer = nn.Linear(embedding_dim, embedding_dim) # 没有使用

        self.ponder_cost = 0.0
        self.sinkhorn_stats = None  # 本层 Sinkhorn 得分矩阵统计量 (batch, 4)
        self.raw_score_stats = None # 本层原始得分矩阵统计量 (batch, 4)，供 Controller 预测 K

        # 两阶段前向的缓存
        self._cached_node_emb = None
        self._cached_mixed_scores = None
        self._cached_v = None

    def prepare_scores(self, node_emb, cost_mat):
        """阶段一：计算 QKV 和原始混合得分矩阵，提取统计量供 Controller 预测 K。

        在已知 K 之前调用。统计量基于 softmax 归一化的原始得分矩阵，
        反映该层注意力得分的合法性（行和偏离1）和确定性（方差/集中度）。

        Returns:
            raw_score_stats: (batch, 4)  — [err_row, err_col, var_row, var_col]
        """
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(node_emb), head_num=head_num)
        k = reshape_by_heads(self.Wk(node_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(node_emb), head_num=head_num)

        mixed_scores = self.mixed_score_MHA.compute_raw_scores(q, k, cost_mat)
        self.raw_score_stats = self.mixed_score_MHA.raw_score_stats

        # 缓存中间结果供 execute_with_K 使用
        self._cached_node_emb = node_emb
        self._cached_mixed_scores = mixed_scores
        self._cached_v = v

        return self.raw_score_stats

    def execute_with_K(self, K_samples):
        """阶段二：使用 Controller 预测的 K 执行 Sinkhorn 并完成该层的剩余计算。

        Args:
            K_samples: (batch,) signed Sinkhorn step counts

        Returns:
            node_emb: (batch, row_cnt, embedding)
        """
        node_emb = self._cached_node_emb
        mixed_scores = self._cached_mixed_scores
        v = self._cached_v

        out_concat = self.mixed_score_MHA.forward_with_scores(mixed_scores, v, K_samples)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        # 获取 Sinkhorn 收敛后的统计量（用于监控和下一层的额外特征）
        self.sinkhorn_stats = self.mixed_score_MHA.sinkhorn_stats

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        # 思考成本：当前层的平均 Sinkhorn 迭代步数
        self.ponder_cost = K_samples.abs().float().mean().item()

        out1 = self.add_n_normalization_1(node_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, row_cnt, embedding)

    def forward(self, node_emb, cost_mat, K_samples):
        """完整的前向传播（向后兼容，用于不需要两阶段分离的场景）。"""
        self.prepare_scores(node_emb, cost_mat)
        return self.execute_with_K(K_samples)



########################################
# Decoder
########################################

class ATSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_0 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention
        self.q1 = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (batch, job, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_jobs), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs), head_num=head_num)
        # shape: (batch, head_num, job, qkv_dim)
        self.single_head_key = encoded_jobs.transpose(1, 2)
        # shape: (batch, embedding, job)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_q0, ninf_mask):
        # encoded_q4.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q1 + q0
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = self._multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, job)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, job)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, job)

        return probs

    def _multi_head_attention(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or pomo
        # k,v shape: (batch, head_num, node, key_dim)
        # rank2_ninf_mask.shape: (batch, node)
        # rank3_ninf_mask.shape: (batch, group, node)

        batch_s = q.size(0)
        n = q.size(2)
        node_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, n, node)

        score_scaled = score / sqrt_qkv_dim
        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, node_cnt)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, node_cnt)

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, node)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, key_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, key_dim)

        out_concat = out_transposed.reshape(batch_s, n, head_num * qkv_dim)
        # shape: (batch, n, head_num*key_dim)

        return out_concat


########################################
# NN SUB FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed