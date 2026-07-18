"""
MNIST 手写数字识别 — RCCA + Conv2d 演示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 
from channel_attn.rcca import RCCA


class ConvBlock(nn.Module):
    """Conv2d -> BN -> RCCA -> 残差"""

    def __init__(self, ch, **rcca_kw):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.rcca = RCCA(channels=ch, **rcca_kw)

    def forward(self, x):
        return self.rcca(self.bn(F.gelu(self.conv(x))))


class MNISTNet(nn.Module):
    """输入 (B, 1, 28, 28) -> 10 类"""

    def __init__(self, competition='rank', regularization='shrc'):
        super().__init__()
        # stem
        self.stem = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        # RCCA 块
        self.block1 = ConvBlock(32, L=8, num_heads=4, p=2,
                                competition=competition,
                                regularization=regularization,
                                lambda_shrc=0.01)
        self.pool1 = nn.MaxPool2d(2)  # 14x14

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block2 = ConvBlock(64, L=8, num_heads=4, p=2,
                                competition=competition,
                                regularization=regularization,
                                lambda_shrc=0.01)
        self.pool2 = nn.MaxPool2d(2)  # 7x7

        # head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.bn0(F.gelu(self.stem(x)))
        x = self.pool1(self.block1(x))
        x = self.bn2(F.gelu(self.conv2(x)))
        x = self.pool2(self.block2(x))
        return self.head(x)


def train(model, loader, opt, device, verbose=True, print_every=10):
    """训练一个 epoch，并可选地打印 batch 级进度。

    参数:
      model: nn.Module
      loader: DataLoader
      opt: optimizer
      device: torch.device
      verbose: 是否打印进度（默认 True）
      print_every: 每隔多少个 batch 输出一次（默认 10）

    返回:
      (avg_loss, avg_acc)
    """
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    try:
        num_batches = len(loader)
    except Exception:
        num_batches = 0

    for i, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        batch_n = y.size(0)
        total += batch_n
        correct += (out.argmax(1) == y).sum().item()
        loss_sum += loss.item() * batch_n

        if verbose and num_batches > 0 and (i % print_every == 0 or i == num_batches):
            loss_avg = loss_sum / total if total > 0 else 0.0
            acc_avg = correct / total if total > 0 else 0.0
            # 使用回车覆盖当前行，PowerShell / Linux 终端均支持
            print(f"\r    batch {i}/{num_batches}  loss={loss_avg:.4f}  acc={acc_avg:.3f}", end="", flush=True)

    if verbose and num_batches > 0:
        # 结束 epoch 后换行
        print()

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 数据
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds_train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    try:
        ds_test = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    except RuntimeError:
        # 下载失败则从训练集分拆
        n = len(ds_train)
        ds_train, ds_test = torch.utils.data.random_split(
            ds_train, [int(n * 0.9), n - int(n * 0.9)],
            generator=torch.Generator().manual_seed(42))
        print("  (test set download failed, split from train)")
    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=256, num_workers=0)

    # 四种配置对比
    configs = [
        ('softmax', 'dropout'),
        ('softmax', 'shrc'),
        ('rank',    'dropout'),
        ('rank',    'shrc'),
    ]

    for comp, reg in configs:
        torch.manual_seed(42)
        model = MNISTNet(competition=comp, regularization=reg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)

        print(f"\n{'='*50}")
        print(f"{comp} + {reg}")
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

        for epoch in range(5):
            # 将 verbose 打开以打印每轮的 batch 进度，print_every 可调整
            loss, acc = train(model, dl_train, opt, device, verbose=True, print_every=10)
            scheduler.step()
            test_acc = evaluate(model, dl_test, device)
            print(f"  epoch {epoch+1}:  train_loss={loss:.4f}  "
                  f"train_acc={acc:.3f}  test_acc={test_acc:.3f}")

    print(f"\n{'='*50}")
    print("done")
