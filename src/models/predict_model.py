import argparse
import sys
import logging
import os

import torch
import click

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

from tqdm import tqdm

from utils import *

latest_model = get_latest_model()

@click.command()
@click.option("--model_path", default=latest_model, help='path to model', prompt=True)
@click.option("--data_path", default="data/processed/test.pt", help='path to data', prompt=True)
@click.option("--dev", default=True, help='use dev set for testing')

def predict(model_path, data_path, dev):
    """ Predicts the model on the test set and prints the accuracy.

    Args:
        model_path (str): path to model
        data_path (str): path to data
        dev (bool): use dev set for testing

    Returns:
        None
    """

    logger = logging.getLogger(__name__)

    logger.info('Testing model')
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")

    # Load mnist/data/processed/trainset.pt
    test_dataset = torch.load(data_path)

    print("Testing dataset loaded")
    # print train data size
    print("Testing dataset size: {}".format(len(test_dataset)))

    if dev:
        print("Using dev set for testing, taking first 100 samples")
        test_dataset = torch.load("data/processed/test_dev.pt")
        print("Dev dataset loaded")
        print("Dev dataset size: {}".format(len(test_dataset)))

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = torch.load(model_path)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        accuracy = 0
        for images, labels in tqdm(testloader):

            log_ps = model(images)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    print("Accuracy: ", accuracy.item()/len(testloader)*100)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    

    # Print help message if no arguments are given
    if len(sys.argv) == 1:
        print("Usage: python predicts_model.py [OPTIONS] COMMAND [ARGS]...")
        print("No arguments given.")
        print("Try 'python predict_model.py --help' for help.")
    
    predict()