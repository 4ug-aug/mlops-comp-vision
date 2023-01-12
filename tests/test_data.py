import torch
from src.data.make_dataset import main
from src.data.cifar import cifar
import os


def test_data_shape():

    trainset, testset = 
    assert trainset.data.shape == (len(trainset), 32, 32, 3)
    assert testset.data.shape == (len(testset), 32, 32, 3)

def test_data_size():
    
    assert len(trainset) == 12639
    assert len(testset) == 500
    assert len(valset) == 500

def test_classes_represented():
    

    train_labels =
    test_labels =
    val_labels =

    assert torch.eq(torch.unique(train_labels), torch.tensor(range(10))).all(), f'One of the datasets do not contain all labels'
    assert torch.eq(torch.unique(test_labels), torch.tensor(range(10))).all(), f'One of the datasets do not contain all labels'
    assert torch.eq(torch.unique(val_labels), torch.tensor(range(10))).all(), f'One of the datasets do not contain all labels'





if __name__ == '__main__':
    pass
