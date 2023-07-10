from torchvision.models import resnet18, resnet50

from torch import nn

from models import LitResnet

import torchvision.transforms as transforms


def get_model(
    weights_path: str = None,
    eval: bool = True,
    pretrained: bool = False,
    lr: float = 2e-4,
    _run=None,
):
    net = resnet18(pretrained=pretrained)
    net.fc = nn.Linear(512 * 1, 10)

    # net = resnet50(pretrained=pretrained)
    # net.fc = nn.Linear(512 * 4, 10)

    # Required layer change for 1-dim input
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Load weights, if coming from a trained model
    if weights_path is not None:
        litnet = LitResnet.load_from_checkpoint(weights_path, net=net)
    else:
        litnet = LitResnet(net, lr=lr, _run=_run)

    if eval:
        litnet.eval()

    return litnet


def get_transforms():
    _transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 1290), transforms.InterpolationMode.NEAREST),
            transforms.Normalize(mean=[-80.0], std=[80]),
        ]
    )

    return _transforms
