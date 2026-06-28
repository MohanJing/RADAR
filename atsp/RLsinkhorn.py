import torch
import torch.nn as nn
from torch.distributions import Categorical

class StepController(nn.Module):
    """Per-layer controller that predicts the optimal Sinkhorn iteration count K
    for each encoder layer. Used in Phase 2 (controller training) of the two-stage
    training pipeline.

    K ∈ [-max_steps, -1] ∪ [1, max_steps]  (2*max_steps categories, never 0).
    - K > 0: start normalization from row (dim=-1), iterate K steps
    - K < 0: start normalization from col (dim=-2), iterate |K| steps

    Category mapping (idx → K):
      idx ∈ [0, max_steps-1]       → K = idx - max_steps       (negative)
      idx ∈ [max_steps, 2*max-1]   → K = idx - max_steps + 1   (positive)

    Args:
        embedding_dim: 节点嵌入维度，也是隐向量均值特征的维度
        max_steps: 最大 |K| 值
        extra_dim: 额外特征维度（Sinkhorn 统计量等），默认 0
    """
    def __init__(self, embedding_dim, max_steps=20, extra_dim=0):
        super().__init__()
        self.max_steps = max_steps
        self.extra_dim = extra_dim
        input_dim = embedding_dim + extra_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * max_steps)  # 2*max_steps categories for K ∈ [-max, -1] ∪ [1, max]
        )

    def forward(self, features):
        # features: (batch, embedding_dim + extra_dim)
        logits = self.mlp(features)
        return Categorical(logits=logits)


class RLSinkhorn(nn.Module):
    """Differentiable Sinkhorn normalization with per-instance adaptive iteration count
    and start direction.

    K_samples ∈ [-max_steps, -1] ∪ [1, max_steps]:
    - K > 0: start from row normalization (dim=-1), iterate K steps
    - K < 0: start from col normalization (dim=-2), iterate |K| steps

    Implementation: at each step t, computes BOTH row-norm and col-norm, then uses a
    per-instance mask to select the correct one based on K sign and step parity:
      K > 0 → row at even t, col at odd t   (row-first alternating)
      K < 0 → col at even t, row at odd t   (col-first alternating)
    All instances share a single loop up to max(|K|) steps.
    """
    def __init__(self, max_steps=20):
        super().__init__()
        self.max_steps = max_steps

    def forward(self, scores, K_samples):
        """
        :param scores: (batch, head, row_cnt, col_cnt) raw mixed attention scores
        :param K_samples: (batch,) signed iteration steps, each in [-max, -1] ∪ [1, max]
        :return: (batch, head, row_cnt, col_cnt) doubly-stochastic matrix at step |K_sample|
        """
        batch, head, row_cnt, col_cnt = scores.shape
        device = scores.device

        # Numerical stability: subtract max along the last dimension
        log_P = scores - scores.max(dim=-1, keepdim=True)[0]

        # Gather index = |K| - 1 for all instances (0-based)
        abs_K = K_samples.abs()                        # (batch,)
        gather_idx = abs_K - 1                         # K=±1 → 0, K=±2 → 1, ...
        max_abs_K = abs_K.max().item()

        # 初始化最终结果张量，避免堆叠所有的中间状态
        P_final = torch.empty_like(scores)
        
        # 扩展 gather_idx 用于后续的掩码匹配
        # 形状: (batch, 1, 1, 1)
        target_step_mask = gather_idx.view(batch, 1, 1, 1)

        for t in range(max_abs_K):
            # 依赖 PyTorch 内置 logsumexp 的数值稳定性
            log_P_row = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            log_P_col = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)

            # 决定当前步使用行归一化还是列归一化
            use_row = ((K_samples > 0) & (t % 2 == 0)) | ((K_samples < 0) & (t % 2 == 1))
            use_row = use_row.view(batch, 1, 1, 1)

            log_P = torch.where(use_row, log_P_row, log_P_col)
            
            # 【修复2】：不再保存到 List 并 Stack。
            # 若当前步数 t 等于该样本所需的步数 (gather_idx)，则将其写入 final 结果。
            is_target_step = (t == target_step_mask)
            
            # 使用掩码更新最终结果，转换为指数空间
            P_final = torch.where(is_target_step, log_P.exp(), P_final)

        return P_final
