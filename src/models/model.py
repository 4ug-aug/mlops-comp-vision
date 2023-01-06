import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import timm


class MyAwesomeModel():
    def __init__(self, classes=10):
        self.classes = classes
        pass

    def model(self):
        model_name = "xception"

        timm_model = timm.create_model(model_name, pretrained=False, num_classes=self.classes)

        return timm_model
        

