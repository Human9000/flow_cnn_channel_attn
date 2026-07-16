"""两层 U-Net：en x2 -> attn -> de -> head。

en 到 de/head 的连接直接作为残差相加，不使用独立 ResBlock。
所有卷积均为 padding=0，跳连在时间轴上居中对齐。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 

def align_center(skip, x):
    start = (skip.shape[-1] - x.shape[-1]) // 2
    return skip[..., start:start + x.shape[-1]], x
 
class EnBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3)
        self.bn = nn.BatchNorm1d(out_channels)
        self.down = nn.AvgPool1d(2, 2)

    def forward(self, x):
        skip = F.relu(self.bn(self.conv(x)))
        return self.down(skip), skip


class DeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_channels, out_channels, 3)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, skip):
        x = F.relu(self.bn(self.conv(self.up(x))))
        skip, x = align_center(skip, x)
        return x + skip
 
class UNetAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.en1 = EnBlock(1, 16)
        self.en2 = EnBlock(16, 32)
        self.de1 = DeBlock(32, 32)
        self.de2 = DeBlock(32, 16)
        self.head = nn.Conv1d(16, 4, 1)

    def forward(self, x):
        x, res1 = self.en1(x)
        x, res2 = self.en2(x) 
        x = self.de1(x, res2)
        x = self.de2(x, res1)
        return torch.softmax(self.head(x), dim=1)


def main():
    torch.manual_seed(0)
    model = UNetAttn()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.running_mean.normal_(0, 0.5); m.running_var.uniform_(0.5, 1.5)
            m.weight.data.normal_(1.0, 0.2); m.bias.data.normal_(0, 0.2)
    model.eval()
    dummy = torch.randn(1, 1, 1000)
    with torch.no_grad():
        y = model(dummy)
    print("整段推理:", tuple(dummy.shape), "->", tuple(y.shape))
    torch.onnx.export(model, dummy, "unet.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "time"}, "output": {0: "batch", 2: "time"}},
        opset_version=13, dynamo=False)
    print("已导出 unet.onnx")


if __name__ == "__main__":
    main()
