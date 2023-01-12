import torch

def cifar(dev=False):
    trainset = torch.load(f"data/processed/train{'_dev' if dev else ''}.pt")
    testset = torch.load(f"data/processed/test{'_dev' if dev else ''}.pt")

    return trainset, testset
