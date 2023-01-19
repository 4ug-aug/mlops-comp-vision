# import torch
# from torchvision import datasets, transforms
# import torch.nn.functional as F
import timm
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, model_name="resnet18", classes=10, pretrained=True):
        super().__init__()

        self.model_name = model_name
        self.classes = classes
        self.pretrained = pretrained

        self.cnn = timm.create_model(self.model_name,
                                     pretrained=self.pretrained,
                                     num_classes=self.classes)

    def forward(self, x):
        x = self.cnn(x)
        return x

    def model(self):
        # https://huggingface.co/microsoft/resnet-50
        model_name = "resnet50"

        timm_model = timm.create_model(model_name, pretrained=False, num_classes=self.classes)

        return timm_model
