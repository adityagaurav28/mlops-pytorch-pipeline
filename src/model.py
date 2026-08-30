import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Small 3-block CNN for 32x32 inputs. Fast enough to train on CPU."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Build the classifier named in the training config."""
    architecture = architecture.lower()

    if architecture == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
        # torchvision's ResNet-18 assumes 224x224 ImageNet inputs. CIFAR-10 is
        # 32x32, so swap the 7x7/stride-2 stem for a 3x3/stride-1 conv and drop
        # the max-pool, otherwise the image is downsampled away before layer1.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    raise ValueError(
        f"Unsupported architecture: {architecture!r} (expected 'resnet18' or 'simple_cnn')"
    )
