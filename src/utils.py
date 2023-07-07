from torchvision.models import resnet18

from torch import nn

from models import LitResnet

import torchvision.transforms as transforms

def get_model():
    # Inference stuff
    net = resnet18()

    # Required layer change for 1-dim input
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(512 * 1, 10)

    litnet = LitResnet.load_from_checkpoint(
        'lightning_logs/version_16/checkpoints/epoch=49-step=650.ckpt', net=net)

    litnet.eval()

    return litnet


def get_transforms():
    _transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 1290), transforms.InterpolationMode.NEAREST),
        transforms.Normalize(mean=[-80.0], std=[80])
    ])

    return _transforms