import pytest
import torch

from src.models_lightning.model import MyAwesomeModel


@pytest.mark.parametrize("batch_size", [16, 32, 48, 64])
def test_model_batches(batch_size):
    # expected number of classes
    n_classes = 20
    model = MyAwesomeModel(classes=n_classes)
    trainset = torch.load("data/processed/train_dev.pt")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    imgs, _ = next(iter(trainloader))
    assert model(imgs).shape == torch.Size([batch_size, n_classes])


def test_model_single():
    n_classes = 20
    model = MyAwesomeModel(classes=n_classes)
    trainset = torch.load("data/processed/train_dev.pt")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64)
    imgs, _ = next(iter(trainloader))
    img = imgs[0].view((1, 3, 224, 224))
    assert model(img).shape == torch.Size([1, 20])
