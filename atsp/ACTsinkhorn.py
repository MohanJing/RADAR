import torch
import torch.nn as nn
import torch.nn.functional as F

class ACTSinkhorn(nn.Module):
    def __init__(self, max_steps=20, eps=1e-3):
        super().__init__()
        self.max_steps = max_steps
        self.eps = eps
        
        # 停止网络：输入行和/列和的平均绝对误差 (2维特征)，输出当前步的停止概率 p_t
        self.halting_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.ponder_cost = 0.0

    def forward(self, scores):
        batch, head, row_cnt, col_cnt = scores.shape
        device = scores.device
        
        # 为防止下溢，先减去最大值（不影响 Softmax 相对概率）
        log_P = scores - scores.max(dim=-1, keepdim=True)[0]
        
        c_t = torch.zeros(batch, head, 1, 1, device=device)
        stopped = torch.zeros(batch, head, 1, 1, dtype=torch.bool, device=device)
        T_idx = torch.zeros(batch, head, 1, 1, dtype=torch.long, device=device)
        
        w_soft_list = []
        P_states_list = []
        ponder_cost_batch = torch.zeros(batch, head, 1, 1, device=device)
        
        for t in range(self.max_steps):
            # 交替进行行/列操作：偶数步作行归一化，奇数步作列归一化
            if t % 2 == 0:
                log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            else:
                log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
                
            P = log_P.exp()
            
            # 特征提取
            # 提取残差特征 (合法性)
            # 沿列(dim=-1)求和得到行和 -> (batch, head, row_cnt, 1)，沿行(dim=-2)求均值 -> (batch, head, 1, 1)
            err_row = (P.sum(dim=-1, keepdim=True) - 1.0).abs().mean(dim=-2, keepdim=True)
            # 沿行(dim=-2)求和得到列和 -> (batch, head, 1, col_cnt)，沿列(dim=-1)求均值 -> (batch, head, 1, 1)
            err_col = (P.sum(dim=-2, keepdim=True) - 1.0).abs().mean(dim=-1, keepdim=True)

            # 提取方差特征 (确定性)
            # 沿列(dim=-1)求方差 -> (batch, head, row_cnt, 1)，沿行(dim=-2)求均值 -> (batch, head, 1, 1)
            var_row = P.var(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            # 沿行(dim=-2)求方差 -> (batch, head, 1, col_cnt)，沿列(dim=-1)求均值 -> (batch, head, 1, 1)
            var_col = P.var(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)

            # 组合 4 维特征
            features = torch.cat([err_row, err_col, var_row, var_col], dim=-1) # shape: (batch, head, 1, 4)
            
            # 计算停止概率 p_t
            p_t = self.halting_mlp(features) # (batch, head, 1, 1)
            
            # 如果到达最大步数，强制令 p_t = 1
            if t == self.max_steps - 1:
                p_t = torch.ones_like(p_t)
            
            # ACT 截断逻辑
            reached_threshold = (c_t + p_t) >= (1.0 - self.eps)
            just_stopped = reached_threshold & (~stopped)
            
            # 记录首次跨过阈值的步数索引 (T)
            T_idx[just_stopped] = t
            
            # 计算软权重
            w_t = torch.where(stopped, torch.zeros_like(p_t),
                  torch.where(reached_threshold, 1.0 - c_t, p_t))
                  
            c_t = c_t + w_t
            stopped = stopped | reached_threshold
            
            w_soft_list.append(w_t)
            P_states_list.append(P)
            
            # 累加思考成本 (Ponder Cost)：1 - c_t 的累加在数学上严格等于期望的迭代步数
            ponder_cost_batch += (1.0 - c_t)
            
            # 如果 batch 内所有样本的所有 head 都停止了，可以提前跳出循环，节省显存
            if stopped.all():
                break
                
        # 拼接张量，shape: (T_max, batch, head, ...)
        w_soft = torch.stack(w_soft_list, dim=0) # (T_max, batch, head, 1, 1)
        P_states = torch.stack(P_states_list, dim=0) # (T_max, batch, head, row_cnt, col_cnt)
        
        # 构造 Hard Weights (One-hot 向量，只有停止步 T 处为 1)
        T_max = w_soft.shape[0]
        time_idx = torch.arange(T_max, device=device).view(-1, 1, 1, 1, 1)
        W_hard = (time_idx == T_idx.unsqueeze(0)).float()
        
        # 【核心魔法：STE 直通估计器】
        # 前向传播：W_final 完全等于 W_hard (只有最后一步为1，不混入未收敛状态)
        # 反向传播：梯度通过 w_soft 回传给 Halting MLP
        W_final = W_hard.detach() - w_soft.detach() + w_soft
        
        # 结合 STE 权重与所有历史状态
        P_final = (W_final * P_states).sum(dim=0)
        
        # 记录当前层的平均思考成本
        self.ponder_cost = ponder_cost_batch.mean()
        
        return P_final