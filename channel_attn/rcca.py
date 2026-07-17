import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================
# RCCA + SHRC: 等级竞争与自修复冗余编码
# 基于 channel_attn/theory.md 第 6-7 节
#
# 双开关:
#   competition   = 'softmax' | 'rank'     — 竞争机制
#   regularization = 'dropout' | 'shrc'    — 互补正则化机制
#
# SHRC 作用于 scores (N,h,L,L), 同时覆盖 h 和 L 两个维度
# ==============================================================


class InjectSHRC(torch.autograd.Function):
    """
    SHRC 覆盖梯度注入, 作用于 scores (N,h,L,L)。

    合并 h*L 为统一实体维度 → 任意两个实体之间最近邻覆盖,
    同时涵盖跨头、跨查询、跨两者的组合。
    """

    @staticmethod
    def forward(ctx, scores, lambda_shrc, n_heads, L):
        ctx.save_for_backward(scores)
        ctx.lambda_shrc = lambda_shrc
        ctx.n_heads = n_heads
        ctx.L = L
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        scores, = ctx.saved_tensors
        h, L = ctx.n_heads, ctx.L
        M = h * L
        if h <= 1:
            return grad_output, None, None, None,None
        with torch.enable_grad():
            s = scores.detach().requires_grad_(True)
            N = s.shape[0]
            s_r = s.reshape(N, M, L)
            # (N, M, 1, L) - (N, 1, M, L) -> (N, M, M, L)
            d = (s_r.unsqueeze(2) - s_r.unsqueeze(1)).pow(2).mean(dim=-1)
            eye = torch.eye(M, device=s.device, dtype=torch.bool)
            d = d.masked_fill(eye.view(1, M, M), float('inf'))
            nearest = d.min(dim=2).values  # (N, M)
            shrc = (nearest.sum() / N) * ctx.lambda_shrc
            grad_aux = torch.autograd.grad(shrc, s, torch.ones_like(shrc))[0]

        return grad_output + grad_aux, None, None, None, None


class RankAttention(nn.Module):
    """等级竞争激活 (6.2): sort -> rank -> C_k = (n-k)^p/n^p -> *P_k"""

    def __init__(self, n_groups, p=2):
        super().__init__()
        self.n = n_groups
        self.p = p
        self.P_k = nn.Parameter(torch.ones(n_groups))

    def forward(self, scores):
        _, order = torch.sort(scores, dim=-1, descending=True)
        rank_idx = torch.argsort(order, dim=-1)
        k = rank_idx.detach().float()
        C_k = ((self.n - k) ** self.p) / (self.n ** self.p)
        return C_k * self.P_k[rank_idx]


class RCCA(nn.Module):
    """
    RCCA: 等级竞争 + SHRC 自修复编码
    (B, C, H, W) -> (B, C, H, W), C = L * F
    """

    def __init__(self, 
                 channels, 
                 L=8,
                 num_heads=1,
                 qk_dim=None, 
                 p=2,
                 competition='rank', 
                 regularization='shrc',
                 0lambda_shrc=0.01, 
                 dropout_rate=0.1):
        super().__init__()
        assert channels % L == 0
        assert competition in ('softmax', 'rank')
        assert regularization in ('dropout', 'shrc')

        self.L = L
        self.F = channels // L
        self.num_heads = num_heads
        self.p = p
        self.competition = competition
        self.regularization = regularization
        self.lambda_shrc = lambda_shrc
        self.dropout_rate = dropout_rate

        if qk_dim is None:
            qk_dim = self.F
        assert qk_dim % num_heads == 0 and self.F % num_heads == 0
        self.qk_dim = qk_dim
        self.d_k = qk_dim // num_heads
        self.d_v = self.F // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(self.F, qk_dim, bias=False)
        self.W_k = nn.Linear(self.F, qk_dim, bias=False)
        self.W_v = nn.Linear(self.F, self.F,  bias=False)
        self.W_o = nn.Linear(self.F, self.F,  bias=False)

        if competition == 'rank':
            self.rank_attn = RankAttention(n_groups=L, p=p)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C, H, W = x.shape
        N = B * H * W

        x_group = x.view(B, self.L, self.F, H, W)
        x_flat = x_group.permute(0, 3, 4, 1, 2).reshape(N, self.L, self.F)

        Q = self.W_q(x_flat)
        K = self.W_k(x_flat)
        V = self.W_v(x_flat)

        Q = Q.view(N, self.L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(N, self.L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(N, self.L, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (N,h,L,L)
        scores = self.dropout(scores)

        # ---- SHRC 作用于 scores (N,h,L,L), 同时覆盖 h 和 L 做特征互补 ----
        if self.training and self.regularization == 'shrc' and self.num_heads > 1:
            scores = InjectSHRC.apply(scores, self.lambda_shrc, self.num_heads, self.L)

        # ---- 竞争 ----
        if self.competition == 'softmax':
            attn_weights = torch.softmax(scores, dim=-1)
        else:
            flat = scores.reshape(N * self.num_heads * self.L, self.L)
            attn_weights = self.rank_attn(flat)
            attn_weights = attn_weights.reshape(N, self.num_heads, self.L, self.L)

        # ---- V 加权 + 合并多头 + 投影 ----
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(N, self.L, self.F)
        out = self.W_o(out)

        out = out.reshape(B, H, W, self.L, self.F)
        out = out.permute(0, 3, 4, 1, 2).reshape(B, C, H, W)
        return out + x

    def set_lambda(self, shrc=None):
        if shrc is not None:
            self.lambda_shrc = shrc


# ==============================================================
# 测试
# ==============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    for comp in ('softmax', 'rank'):
        for reg in ('dropout', 'shrc'):
            nh = 4 if reg == 'shrc' else 1
            model = RCCA(channels=256, L=8, num_heads=nh,
                         competition=comp, regularization=reg,
                         lambda_shrc=0.01)
            model.train()

            x = torch.randn(2, 256, 32, 32)
            out = model(x)
            loss = F.mse_loss(out, torch.randn_like(out))
            loss.backward()

            n_grad = sum(1 for p in model.parameters()
                         if p.grad is not None and p.grad.norm() > 0)
            print(f"{comp:>8} + {reg:>7}  h={nh}"
                  f"  loss={loss.item():.4f}  grads={n_grad}  [OK]")

    # 梯度注入验证
    torch.manual_seed(42)
    m = RCCA(channels=256, L=8, num_heads=4,
             competition='rank', regularization='shrc', lambda_shrc=0.1)

    x = torch.randn(2, 256, 32, 32); t = torch.randn(2, 256, 32, 32)

    m.eval(); o = m(x); F.mse_loss(o, t).backward()
    g_task = m.W_o.weight.grad.clone(); m.zero_grad()

    m.train(); o = m(x); F.mse_loss(o, t).backward()
    g_both = m.W_o.weight.grad.clone()

    print(f"\ntask_grad: {g_task.norm().item():.4f}")
    print(f"both_grad: {g_both.norm().item():.4f}")
    print(f"aux_grad:  {(g_both - g_task).norm().item():.4f}  (>0 SHRC OK)")
    print(f"\nparams: {sum(p.numel() for p in m.parameters()):,}")
    print("OK  all 4 combos passed")
