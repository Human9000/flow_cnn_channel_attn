import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    自实现多头注意力（batch_first）
    - 支持 Q/K 压缩
    - 支持 softmax / sigmoid 切换
    - 输入形状: (N, S, E) → 输出形状: (N, S, E)
    """

    def __init__(self, embed_dim, num_heads=4, qk_dim=None,
                 dropout_rate=0.1, activation='softmax'):
        super().__init__()
        assert activation in ('softmax', 'sigmoid', 'competition', 'simple', 'sparse'), (
            f"activation 必须是 'softmax'、'sigmoid'、'competition'、'simple' 或 'sparse'，收到: {activation}")
        assert embed_dim % num_heads == 0, (
            f"embed_dim({embed_dim}) 必须能被 num_heads({num_heads}) 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.activation = activation

        if qk_dim is None:
            qk_dim = embed_dim
        assert qk_dim % num_heads == 0, (
            f"qk_dim({qk_dim}) 必须能被 num_heads({num_heads}) 整除")

        self.qk_dim = qk_dim
        self.d_k = qk_dim // num_heads
        self.d_v = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, qk_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, qk_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x):
        N, S, E = x.shape

        Q = self.W_q(x)  # (N, S, qk_dim)
        K = self.W_k(x)  # (N, S, qk_dim)
        V = self.W_v(x)  # (N, S, E)

        Q = Q.view(N, S, self.num_heads, self.d_k).transpose(1, 2)  # (N, h, S, d_k)
        K = K.view(N, S, self.num_heads, self.d_k).transpose(1, 2)  # (N, h, S, d_k)
        V = V.view(N, S, self.num_heads, self.d_v).transpose(1, 2)  # (N, h, S, d_v)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (N, h, S, S)
        scores = self.dropout(scores)  # 在归一化之前做，sum=1 不被破坏

        if self.activation == 'softmax':
            attn_weights = torch.softmax(scores, dim=-1)
        elif self.activation == 'sigmoid':
            attn_weights = torch.sigmoid(scores)
        elif self.activation == 'competition':
            # 标准化 → sigmoid → 平方 → 概率密度化
            eps = 1e-6
            mean = scores.mean(dim=-1, keepdim=True)  
            g = torch.sigmoid(scores - mean)
            g2 = g * g
            attn_weights = g2 / (g2.sum(dim=-1, keepdim=True) + eps)
        elif self.activation == 'simple':
            # 归一化 → 平方 → 概率密度化
            eps = 1e-6
            s_min = scores.min(dim=-1, keepdim=True).values
            s_max = scores.max(dim=-1, keepdim=True).values
            normed = (scores - s_min) / (s_max - s_min + eps)
            sq = normed * normed
            attn_weights = sq / (sq.sum(dim=-1, keepdim=True) + eps)
        else:  
            # sparse: 分段线性稀疏注意力 — 模拟 softmax exp(-4) 截断
            # exp(-4) ≈ 1.8%，即比 max 小 4 以上的元素在 softmax 中可忽略
            # 用 relu(s - max + 4) 线性近似 softmax，窗口外显式归零
            eps = 1e-6
            s_max = scores.max(dim=-1, keepdim=True).values
            x01 = torch.relu(scores - s_max + 2) / 2
            x01 = x01 * x01
            attn_weights = x01 / (x01.sum(dim=-1, keepdim=True) + eps)

        out = torch.matmul(attn_weights, V)  # (N, h, S, d_v)
        out = out.transpose(1, 2).contiguous().view(N, S, E)
        out = self.W_o(out)

        return out


class GroupMHA(nn.Module):
    """
    通道分组自注意力
    - 将 B, H, W 合并为 Batch 维度
    - 将 C 拆分为 L（序列长度）和 F（嵌入维度）
    - 每个像素独立的 L 组竞争/互补
    """

    def __init__(self, channels, L=8, num_heads=1, qk_dim=None,
                 dropout_rate=0.1, activation='softmax'):
        super().__init__()
        assert channels % L == 0, "channels 必须能被 L 整除"

        self.L = L
        self.F = channels // L

        self.attn = MultiHeadAttention(
            embed_dim=self.F,
            num_heads=num_heads,
            qk_dim=qk_dim,
            dropout_rate=dropout_rate,
            activation=activation,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, C, H, W) → (B*H*W, L, F)
        x_group = x.view(B, self.L, self.F, H, W)
        x_flat = x_group.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.L, self.F)

        # 自注意力
        out = self.attn(x_flat)  # (B*H*W, L, F)

        # (B*H*W, L, F) → (B, C, H, W)
        out = out.reshape(B, H, W, self.L, self.F)
        out = out.permute(0, 3, 4, 1, 2).reshape(B, C, H, W)

        return out + x