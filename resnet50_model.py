"""timm ImageNet-1K 预训练 ResNet-50 基线模型。"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError as error:
    raise ImportError("需要安装 timm：python -m pip install timm") from error


MODEL_NAME = "resnet50.a1_in1k"
INPUT_SIZE = (3, 224, 224)
INPUT_MEAN = (0.485, 0.456, 0.406)
INPUT_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 1000


class ResNet50(nn.Module):
    """保持普通 nn.Module 接口的 timm 预训练 ResNet-50。"""

    def __init__(self, pretrained=True):
        super().__init__()
        self.network = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            exportable=True,
        )

    def forward(self, image):
        return self.network(image)


def normalize_image(image):
    mean = image.new_tensor(INPUT_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(INPUT_STD).view(1, 3, 1, 1)
    return (image - mean) / std


if __name__ == "__main__":
    model = ResNet50().eval()
    image = torch.zeros(1, *INPUT_SIZE)
    with torch.no_grad():
        logits = model(normalize_image(image))
    print(MODEL_NAME, tuple(image.shape), "->", tuple(logits.shape))
