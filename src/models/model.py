import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import timm


class MyAwesomeModel(nn.Module):
    def __init__(self):
        model_name = "xception"

        timm_model = timm.create_model(model_name, pretrained=True)

        return timm_model
        

