from torchvision.models import (
    resnet18,
    resnet50,
    efficientnet_b4,
    efficientnet_b0,
    efficientnet_b1,
    EfficientNet_B1_Weights,
    EfficientNet_B4_Weights,
)


from data import NpyDataset

from torch import nn
import torch

import numpy as np

from collections import OrderedDict

from models import LitResnet
import lightning.pytorch as pl

import torchvision.transforms as transforms

from data_augmentation import get_n_variations


def get_dataset(
    dataset: str,
    root_data_folder: str = "data",
    metadata_file_name: str = "metadata.csv",
    subset: str = "train",
    original_only: bool = False,
    subset_mode: str = "default",
):
    """
    Utility function to retrieve a specific subset of a
    specific version of the dataset.

    dataset: str
        The version of the dataset
    original_only: bool
        Whether to return only the "original" (i.e. non-augmented)
        versions of the audio clips.
    """

    train_transforms = get_transforms()

    _dataset = NpyDataset(
        f"{root_data_folder}/{dataset}",
        f"{root_data_folder}/{dataset}/{metadata_file_name}",
        subset,
        transform=train_transforms,
        original_only=original_only,
        subset_mode=subset_mode,
    )

    return _dataset


def get_datasets(
    dataset: str,
    root_data_folder: str = "data",
    metadata_file_name: str = "metadata.csv",
    train_subset: str = "train",
    test_subset: str = "test",
    original_only: bool = False,
):
    dataset_train = get_dataset(
        dataset,
        root_data_folder=root_data_folder,
        metadata_file_name=metadata_file_name,
        subset=train_subset,
        original_only=original_only,
    )
    dataset_test = get_dataset(
        dataset,
        root_data_folder=root_data_folder,
        metadata_file_name=metadata_file_name,
        subset=test_subset,
        original_only=original_only,
    )

    return dataset_train, dataset_test


# def get_resnet18(
#     weights_path: str = None,
#     eval: bool = True,
#     lr: float = 2e-4,
#     lr_scheduler_gamma: float = 0.95,
#     _run=None,
# ) -> pl.LightningModule:
#     net = resnet18(pretrained=pretrained)

#     # Small hack to allow 1-channel input into efficientnet, plus the use of
#     # pretrained weights
#     net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     net.fc = nn.Linear(512 * 1, 10)

#     # Load weights, if coming from a trained model
#     if weights_path is not None:
#         litnet = LitResnet.load_from_checkpoint(weights_path, net=net)
#     else:
#         litnet = LitResnet(net, lr=lr, _run=_run)

#     if eval:
#         litnet.eval()

#     return litnet


def get_effnet_b1(
    weights_path: str = None,
    eval: bool = True,
    lr: float = 2e-4,
    lr_scheduler_gamma: float = 0.95,
    _run=None,
) -> pl.LightningModule:
    net = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)

    # Small hack to allow 1-channel input into efficientnet, plus the use of
    # pretrained weights
    net.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=3, stride=2, padding=1, bias=False
    )
    net.classifier[1] = nn.Linear(1280, 10, bias=True)

    # Load weights, if coming from a trained model
    if weights_path is not None:
        litnet = LitResnet.load_from_checkpoint(weights_path, net=net)
    else:
        litnet = LitResnet(net, lr=lr, lr_scheduler_gamma=lr_scheduler_gamma, _run=_run)

    if eval:
        litnet.eval()

    return litnet


def get_transforms():
    _transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 1290), transforms.InterpolationMode.NEAREST),
            transforms.Normalize(
                mean=[-72.79], std=[12.02]
            ),  # computed from training data
        ]
    )

    return _transforms


# TODO change output type
def sort_votes(votes: np.ndarray, return_all: bool = False) -> OrderedDict:
    """
    Counts and sorts votes for all categories.

    votes: np.ndarray:
        Contains the categories predicted on the different augmentations

    return_all: bool
        Return the full sorted list, for a more complete overview of predictions.
    """

    sorted_votes = {}

    for v in np.unique(votes):
        sorted_votes[v] = (votes == v).sum()

    sv_tuples = sorted(sorted_votes.items(), key=lambda x: x[1], reverse=True)

    sv_dict = OrderedDict(sv_tuples)

    if return_all:
        return sv_tuples[0][0], sv_dict
    else:
        return sv_tuples[0][0]


def inference_one(
    audio_file_path: str, n_augmentations: int, model: pl.LightningModule
) -> int:
    variation_db_mels = get_n_variations(
        audio_file_path, n_augmentations, return_original=True
    )
    """
    Performs inference on a single audio file, returns category index (0-9)
    """

    train_transforms = get_transforms()

    # TODO change this to allow non-cuda
    input_tensors = torch.stack([train_transforms(x) for x in variation_db_mels]).to(
        "cuda"
    )

    y_hat = model(input_tensors)

    cat_idx = torch.argmax(y_hat, dim=1)

    votes = cat_idx.cpu().numpy()

    majority_category = sort_votes(votes)

    return majority_category
