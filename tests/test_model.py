from src.models.model import MyAwesomeModel
from src.data.cifar import cifar 
import torch
import pytest

@pytest.mark.parametrize("batch_size", [16,32,64,128])
def test_model_batches(batch_size):
    model = MyAwesomeModel()
    trainset, _ = cifar()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    imgs, _ = next(iter(trainloader))
    assert model(imgs).shape == torch.Size([batch_size,10])

def test_model_single():
    model = MyAwesomeModel()
    trainset, _ = cifar()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64)
    imgs, _ = next(iter(trainloader))
    img = imgs[0].view((1,3,32,32))
    assert model(img).shape == torch.Size([1,10])