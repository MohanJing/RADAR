import torch
import torch.nn as nn
from torch.distributions import Categorical

class StepController(nn.Module):
    """Graph-level controller that predicts the optimal number of Sinkhorn iterations K
    for each problem instance. Used in Phase 2 (controller training) of the two-stage
    training pipeline."""
    def __init__(self, embedding_dim, max_steps=20):
        super().__init__()
        self.max_steps = max_steps
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_steps)  # output logits for M categories (K=1..max_steps)
        )

    def forward(self, graph_emb):
        # graph_emb: (batch, embedding_dim)
        logits = self.mlp(graph_emb)
        return Categorical(logits=logits)


class RLSinkhorn(nn.Module):
    """Differentiable Sinkhorn normalization with adaptive iteration count.

    Runs Sinkhorn up to max(K_samples) steps, then gathers the doubly-stochastic
    matrix at step K_sample for each instance in the batch. This enables the model
    to learn instance-adaptive computation budgets.

    K_samples is predicted by StepController (Phase 2) or sampled uniformly (Phase 1).
    """
    def __init__(self, max_steps=20):
        super().__init__()
        self.max_steps = max_steps

    def forward(self, scores, K_samples):
        """
        :param scores: (batch, head, row_cnt, col_cnt) raw mixed attention scores
        :param K_samples: (batch,) discrete iteration steps, each in [1, max_steps]
        :return: (batch, head, row_cnt, col_cnt) doubly-stochastic matrix at step K_sample
        """
        batch, head, row_cnt, col_cnt = scores.shape

        # Numerical stability: subtract max along the last dimension
        log_P = scores - scores.max(dim=-1, keepdim=True)[0]

        # Run Sinkhorn iterations up to the maximum K in this batch
        max_k = K_samples.max().item()
        P_states = []

        for t in range(max_k):
            # Alternating row/column normalization in log-space
            if t % 2 == 0:
                log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            else:
                log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
            P_states.append(log_P.exp())

        # Stack all intermediate states: (max_k, batch, head, row_cnt, col_cnt)
        P_stack = torch.stack(P_states, dim=0)

        # For each instance, gather the doubly-stochastic matrix at its K_sample-th step
        # K_samples values are in [1, max_steps]; subtract 1 for 0-based indexing into P_stack
        K_idx = (K_samples - 1).view(1, batch, 1, 1, 1) \
                               .expand(1, batch, head, row_cnt, col_cnt) \
                               .to(device=scores.device)

        P_final = torch.gather(P_stack, dim=0, index=K_idx).squeeze(0)
        # shape: (batch, head, row_cnt, col_cnt)

        return P_final
