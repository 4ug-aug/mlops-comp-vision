import argparse
import sys
import logging
import os

import torch
import click
from torchvision import transforms

# Timm computer vision hugging face library
import timm
from timm.optim.optim_factory import create_optimizer
from timm.data import create_dataset, create_loader
from types import SimpleNamespace

from model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from tqdm import tqdm

from utils import count_files

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=1, help='number of epochs to train for')
@click.option("--dev", default=True, help='use dev set for training')

# Make help message for the train command
@click.help_option("--help", "-h")

def train(lr, epochs, dev):
    logger = logging.getLogger(__name__)

    logger.info('training model')
    logger.info(f"Learning Rate: {lr}")
    logger.info(f"Epochs: {epochs}")

    # Load mnist/data/processed/trainset.pt
    train_dataset = torch.load("data/processed/train.pt")

    print("Training dataset loaded")
    # print train data size
    print("Training dataset size: {}".format(len(train_dataset)))

    if dev:
        print("Using dev set for training, taking first 100 samples")
        train_dataset = torch.load("data/processed/train_dev.pt")
        print("Dev dataset loaded")
        print("Dev dataset size: {}".format(len(train_dataset)))

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = MyAwesomeModel(classes=10).model()

    criterion = nn.CrossEntropyLoss()

    # Create optimiser
    args = SimpleNamespace()
    args.weight_decay = 0
    args.lr = lr
    args.opt = 'adam' #'lookahead_adam' to use `lookahead`
    args.momentum = 0.9

    optimizer = create_optimizer(args, model)

    training_loss = []

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        running_loss = 0
        for images, labels in tqdm(trainloader):
            
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
        else:
            training_loss.append(running_loss/trainloader.batch_size)
            print(f"Training loss: {training_loss[-1]}")

    # plot the training loss
    plt.plot(training_loss, label='Training loss')
    plt.legend(frameon=False)

    new_path = count_files("reports/figures")

    # Save plot
    plt.savefig(f"reports/figures/training_loss_{new_path}.png")

    # Generate unique name for model
    new_path = count_files("src/models/trained_models")
    new_path = f"models/trained_models/model_checkpoint_{new_path}.pth"

    print("Saving model as model_checkpoint.pth")
    print("Path: {}".format(new_path))
    torch.save(model, new_path)


cli.add_command(train)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Print help message if no arguments are given
    if len(sys.argv) == 1:
        print("Usage: python train_model.py [OPTIONS] COMMAND [ARGS]...")
        print("No arguments given, using default values")
        print("Default values: --lr 0.001 --epochs 7")

    train()