import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import numpy as np

def PIL_to_tensor(input_filepath, imgs_PIL):
    transform = transforms.Compose([transforms.PILToTensor()])
    train_imgs = []
    for image_path in tqdm(imgs_PIL):
        img = Image.open(f"{input_filepath }/{image_path}")
        train_imgs.append(transform(img).unsqueeze(dim=0))

    train_imgs = torch.cat(train_imgs)
    return train_imgs