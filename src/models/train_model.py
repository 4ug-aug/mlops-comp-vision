import argparse
import sys
import logging
import os

import torch
import click

from model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

from utils import count_files

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=7, help='number of epochs to train for')

# Make help message for the train command
@click.help_option("--help", "-h")

def train(lr, epochs):
    logger = logging.getLogger(__name__)

    logger.info('training model')
    logger.info(f"Learning Rate: {lr}")
    logger.info(f"Epochs: {epochs}")

    model = MyAwesomeModel()

    # Load mnist/data/processed/trainset.pt
    train_set = torch.load("data/processed/training.pt")
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    
    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_loss = []

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        running_loss = 0
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            training_loss.append(running_loss/len(trainloader))
            print(f"Training loss: {training_loss[-1]}")

    # plot the training loss
    plt.plot(training_loss, label='Training loss')
    plt.legend(frameon=False)

    new_path = count_files("reports/figures")

    # Save plot
    plt.savefig(f"reports/figures/training_loss_{new_path}.png")

    # Generate unique name for model
    new_path = count_files("src/models/trained_models")
    new_path = f"src/models/trained_models/model_checkpoint_{new_path}.pth"

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