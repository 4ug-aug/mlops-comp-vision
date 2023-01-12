import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import timm


class MyAwesomeModel(nn.Module):
    def __init__(self, model_name="resnet18",classes=10, pretrained=True):
        super().__init__()

        self.model_name = model_name
        self.classes = classes
        self.pretrained = pretrained

        self.cnn = timm.create_model(self.model_name,pretrained=self.pretrained,num_classes = self.classes)

        self.fc1 = nn.Linear(3*32*32, 3*224*224)
    
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = x.view((x.size(0),3,224,224))
        x = self.cnn(x)
        return x
        


    def model(self):
        # https://huggingface.co/microsoft/resnet-50
        model_name = "resnet50"

        timm_model = timm.create_model(model_name, pretrained=False, num_classes=self.classes)

        return timm_model
    

        

