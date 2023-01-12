import torch
from hydra.utils import get_original_cwd

def cifar(dev=False):
    trainset = torch.load(f"{get_original_cwd()}/data/processed/train{'_dev' if dev else ''}.pt")
    testset = torch.load(f"{get_original_cwd()}/data/processed/test{'_dev' if dev else ''}.pt")

    if dev:
        print("Using dev set for training, taking first 5000 samples")
        print("Dev dataset loaded")
        print("Dev dataset size: {}".format(len(trainset)))
    else:
        print("Training dataset loaded")
        # print train data size
        print("Training dataset size: {}".format(len(trainset)))

    return trainset, testset
