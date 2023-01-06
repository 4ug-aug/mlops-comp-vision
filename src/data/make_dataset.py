# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import torch
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from ('~/.pytorch/MNIST_data/') into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    X_train = np.load('data/raw/train_0.npz')['images']
    y_train = np.load('data/raw/train_0.npz')['labels']

    for i in range(1, 5):
        X_train = np.concatenate((X_train, np.load(f'data/raw/train_{i}.npz')['images']), axis=0)
        y_train = np.concatenate((y_train, np.load(f'data/raw/train_{i}.npz')['labels']), axis=0)
        
    
    X_test = np.load('data/raw/test.npz')["images"]
    y_test = np.load('data/raw/test.npz')["labels"]

    # Add labels to the test set
    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    # Add labels to the train set
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())

    # Save the data
    torch.save(train, 'data/processed/training.pt')
    torch.save(test, 'data/processed/test.pt')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
