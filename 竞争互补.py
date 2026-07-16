import torch
import torch.nn as nn

class GroupMHA(nn.Module):
    """
    纯通道分组 MHA
    - 将 B, H, W 合并为 Batch 维度
    - 将 C 拆分为 L（序列长度）和 F（嵌入维度）
    - 每个像素独立的 8 组竞争
    """
    def __init__(self, channels, L=8, num_heads=1, dropout_rate=0.1):
        super().__init__()
        assert channels % L == 0, "channels 必须能被 L 整除"
        self.L = L
        self.F = channels // L  # 每组特征维度
        
        # 官方 MHA：embed_dim = F, 序列长度 = L
        # num_heads 必须整除 F。如果 F=256，设 num_heads=1 最纯粹（单头组间注意力）
        # 若想引入多头多样性，可设 num_heads=4（需 F % 4 == 0）
        assert self.F % num_heads == 0, f"F({self.F}) 必须能被 num_heads({num_heads}) 整除"
        self.mha = nn.MultiheadAttention(
            embed_dim=self.F,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # 输入形状: (Batch, Seq, Embed)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # ---- 重塑张量视图 ----
        # 1. 将 C 拆分为 (L, F): (B, L, F, H, W)
        x_group = x.view(B, self.L, self.F, H, W)
        
        # 2. 将 B, H, W 合并为 Batch 维度: (B*H*W, L, F)
        # 这一步后，每个像素位置是一个独立的样本，序列长度为 L（8个组）
        x_flat = x_group.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.L, self.F)
        
        # ---- 标准多头注意力 (组间竞争) ----
        # Q/K/V 均来自 x_flat，即 8 个组之间做自注意力
        # Softmax 在 L 维上竞争，Dropout 强制非独立互补
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        
        # ---- 恢复原始形状 ----
        # 1. 重塑回 (B, H, W, L, F)
        out = attn_out.reshape(B, H, W, self.L, self.F)
        # 2. 转回 (B, L, F, H, W) -> (B, C, H, W)
        out = out.permute(0, 3, 4, 1, 2).reshape(B, C, H, W)
        
        # ---- 标准残差连接 ----
        return out + x