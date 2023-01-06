import argparse
import sys
import logging
import os

import torch
import click

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

from utils import *

@click.group()
def cli():
    pass

latest_model = get_latest_model()

@click.command()
@click.option("--model_path", default=latest_model, help='path to model', prompt=True)
@click.option("--data_path", default="data/processed/test.pt", help='path to data', prompt=True)
@click.option("--dev", default=True, help='use dev set for testing')

def predict(model_path, data_path, dev):
    logger = logging.getLogger(__name__)

    logger.info('Testing model')
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")

    # Load mnist/data/processed/trainset.pt
    test_dataset = torch.load("data/processed/test_dev.pt")

    print("Testing dataset loaded")
    # print train data size
    print("Testing dataset size: {}".format(len(test_dataset)))

    if dev:
        print("Using dev set for testing, taking first 100 samples")
        test_dataset = torch.load("data/processed/test_dev.pt")
        print("Dev dataset loaded")
        print("Dev dataset size: {}".format(len(test_dataset)))

    test_loader  = create_loader(test_dataset, input_size=(3, 32, 32), batch_size=64, use_prefetcher=False, 
                              is_training=False, no_aug=True, transform=transform)

    model = MyAwesomeModel(classes=10).model()

    criterion = nn.NLLLoss()

    with torch.no_grad():
        accuracy = 0
        for i, (inputs, targets) in tk0:
    
            log_ps = model(inputs)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    print("Accuracy: ", accuracy.item()/len(testloader)*100)


cli.add_command(predict)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    

    # Print help message if no arguments are given
    if len(sys.argv) == 1:
        print("Usage: python predicts_model.py [OPTIONS] COMMAND [ARGS]...")
        print("No arguments given.")
        print("Try 'python predict_model.py --help' for help.")
    
    predict()