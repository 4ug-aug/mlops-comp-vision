import torch
from src.data.make_dataset import main
from src.data.load_data import load_data
import os


def test_data_shape():

    trainset, testset, valset = load_data()
    assert trainset[:][0].shape == (len(trainset), 3, 224, 224)
    assert testset[:][0].shape == (len(testset), 3, 224, 224)
    assert valset[:][0].shaoe == (len(valset), 3, 224, 224)

def test_data_size():
    trainset, testset, valset = load_data()
    assert len(trainset) == 12639
    assert len(testset) == 100
    assert len(valset) == 100

def test_classes_represented():
    
    trainset, testset, valset = load_data()
    train_labels = trainset[:][1]
    test_labels = testset[:][1]
    val_labels = valset[:][1]

    assert torch.eq(torch.unique(train_labels), torch.tensor(range(20))).all(), f'One of the datasets do not contain all labels'
    assert torch.eq(torch.unique(test_labels), torch.tensor(range(20))).all(), f'One of the datasets do not contain all labels'
    assert torch.eq(torch.unique(val_labels), torch.tensor(range(20))).all(), f'One of the datasets do not contain all labels'