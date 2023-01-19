import torch

# import os


def test_data_shape():

    trainset = torch.load("data/processed/train_dev.pt")
    testset = torch.load("data/processed/test.pt")
    valset = torch.load("data/processed/val.pt")
    assert trainset[:][0].shape == (100, 3, 224, 224)
    assert testset[:][0].shape == (100, 3, 224, 224)
    assert valset[:][0].shape == (100, 3, 224, 224)


def test_data_size():
    trainset = torch.load("data/processed/train_dev.pt")
    testset = torch.load("data/processed/test.pt")
    valset = torch.load("data/processed/val.pt")
    assert len(trainset) == 100  # not possible to store full trainset on git
    assert len(testset) == 100
    assert len(valset) == 100


def test_classes_represented():

    trainset = torch.load("data/processed/train_dev.pt")
    testset = torch.load("data/processed/test.pt")
    valset = torch.load("data/processed/val.pt")
    train_labels = trainset[:][1]
    test_labels = testset[:][1]
    val_labels = valset[:][1]

    assert torch.eq(
        torch.unique(train_labels), torch.tensor(range(20))
    ).all(), "One of the datasets do not contain all labels"
    assert torch.eq(
        torch.unique(test_labels), torch.tensor(range(20))
    ).all(), "One of the datasets do not contain all labels"
    assert torch.eq(
        torch.unique(val_labels), torch.tensor(range(20))
    ).all(), "One of the datasets do not contain all labels"
