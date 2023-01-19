import torch


def load_data():
    trainset = torch.load("data/processed/train.pt")
    testset = torch.load("data/processed/test.pt")
    valset = torch.load("data/processed/val.pt")

    return trainset, testset, valset
