import argparse
import sys
import logging
import os
import torch
from torchvision import transforms

# Timm computer vision hugging face library
import timm
from timm.optim.optim_factory import create_optimizer
from timm.data import create_dataset, create_loader
from types import SimpleNamespace

from src.models.model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from tqdm import tqdm
import hydra
from hydra.utils import get_original_cwd

from src.models.utils import count_files

@hydra.main(config_path='conf/',config_name="default_config.yaml")
def main(cfg):
    """ Train a model and save loss plot and model checkpoint

    Args:
        lr (float): learning rate to use for training
        epochs (int): number of epochs to train for
        dev (bool): use dev set for training

    Returns:
        None
    """

    logger = logging.getLogger(__name__)

    cfg = cfg.experiment
    logger.info('Training model')
    logger.info(f"Learning Rate: {cfg.hyperparameters.lr}")
    logger.info(f"Epochs: {cfg.hyperparameters.epochs}")

    if not cfg.hyperparameters.unit_testing:
        trainset = torch.load(f"{get_original_cwd()}/{cfg.hyperparameters.dataset}/train.pt")
    else:
        trainset = torch.load(f"{cfg.hyperparameters.dataset}/val.pt")
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)

    model = MyAwesomeModel(classes=cfg.hyperparameters.n_classes)
    print(cfg.hyperparameters.n_classes)
    criterion = nn.CrossEntropyLoss()


    if cfg.hyperparameters.opt == "adam":
        optimizer = optim.Adam(model.parameters(), 
                               lr=cfg.hyperparameters.lr,
                               weight_decay=cfg.hyperparameters.weight_decay)
    
    training_loss = []

    for e in range(cfg.hyperparameters.epochs):
        print(f"Epoch {e+1}/{cfg.hyperparameters.epochs}")
        running_loss = 0
        for images, labels in tqdm(trainloader):
            
            optimizer.zero_grad()
            
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
        else:
            training_loss.append(running_loss/cfg.hyperparameters.batch_size)
            print(f"Training loss: {training_loss[-1]}")


    if cfg.hyperparameters.unit_testing:
        return running_loss
    
    # plot the training loss
    plt.plot(training_loss, label='Training loss')
    plt.legend(frameon=False)

    new_path = count_files(get_original_cwd() + "/reports/figures")

    # Save plot
    plt.savefig(f"{get_original_cwd()}/reports/figures/training_loss_{new_path}.png")

    # Generate unique name for model
    new_path = count_files(get_original_cwd() + "/models/trained_models")
    new_path = f"{get_original_cwd()}/models/trained_models/model_checkpoint_{new_path}.pth"

    print("Saving model as model_checkpoint.pth")
    print("Path: {}".format(new_path))
    torch.save(model, new_path)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Print help message if no arguments are given
    """
    if len(sys.argv) == 1:
        print("Usage: python train_model.py [OPTIONS] COMMAND [ARGS]...")
        print("No arguments given, using default values")
        print("Default values: --lr 0.001 --epochs 7")
    """

    main()