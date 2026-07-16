import torch
import torch.nn as nn

class GranularSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        squeezed = max(1, channels // reduction)

        self.attention = nn.Sequential(
            # 1. 两次下采样：HxW -> H/4 x W/4 -> H/16 x W/16 (组卷积，保持通道数 C)
            nn.Conv2d(channels, channels, kernel_size=7, stride=4, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=7, stride=4, padding=2, groups=channels, bias=False),

            # 2. 挤压，激活，膨胀，激活
            nn.Conv2d(channels, squeezed, kernel_size=1, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(squeezed, channels, kernel_size=1, bias=False), 
            nn.Sigmoid(),
 
            # 6. 16倍双线性插值上采样回原尺寸
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        return x * self.attention(x)


# ========== 使用示例 ==========
if __name__ == "__main__":
    x = torch.randn(4, 64, 224, 224).cuda()  # 输入必须是 16 的倍数
    model = GranularSE(channels=64, reduction=16).cuda()
    print(x.shape)
    out = model(x)
    print(out.shape)  # torch.Size([4, 64, 224, 224])

