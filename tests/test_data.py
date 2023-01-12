import torch
from src.data.make_dataset import main
from src.data.cifar import cifar
import os


def test_data_shape():

    trainset, testset = cifar()
    assert trainset.data.shape == (len(trainset), 32, 32, 3)
    assert testset.data.shape == (len(testset), 32, 32, 3)
    
    train_dev, test_dev = cifar(dev=True)
    train_dev, test_dev = trainset.data[train_dev.indices], testset.data[test_dev.indices]
    assert train_dev.data.shape == (len(train_dev), 32, 32, 3)
    assert test_dev.data.shape == (len(test_dev), 32, 32, 3)

def test_data_size():

    trainset, testset = cifar()
    assert len(trainset) == 50000
    assert len(testset) == 10000

    train_dev, test_dev = cifar(dev=True)
    assert len(train_dev) == 5000
    assert len(test_dev) == 1000

def test_classes_represented():
    
    trainset, testset = cifar()
    train_dev, test_dev = cifar(dev=True)
    
    train_dev = torch.utils.data.DataLoader(train_dev)
    test_dev = torch.utils.data.DataLoader(test_dev)
    train_dev_targets = torch.tensor([])
    test_dev_targets = torch.tensor([])
    for _, labels in train_dev:
        train_dev_targets = torch.cat([train_dev_targets, labels], dim=0)
    for _ , labels in test_dev:
        test_dev_targets = torch.cat([test_dev_targets, labels], dim=0)
    

    target_list = [torch.tensor(trainset.targets), torch.tensor(testset.targets), train_dev_targets, test_dev_targets]
    for targets in target_list:
        assert torch.eq(torch.unique(targets), torch.tensor(range(10))).all(), f'One of the datasets do not contain all labels'





if __name__ == '__main__':
    pass
