# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import torch
import numpy as np
import timm


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from ('~/.pytorch/MNIST_data/') into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_dataset = timm.data.create_dataset('torch/cifar10','cifar10',download=True,split='train')
    test_dataset = timm.data.create_dataset('torch/cifar10','cifar10',download=True,split='test')

    # Save the data
    torch.save(train_dataset, output_filepath + '/train.pt')
    torch.save(test_dataset, output_filepath + '/test.pt')

    # make dev set
    train_indices = torch.randperm(len(train_dataset))[:5000]
    train_dataset_dev = torch.utils.data.Subset(train_dataset, train_indices)

    test_indices = torch.randperm(len(test_dataset))[:1000]
    test_dataset_dev = torch.utils.data.Subset(test_dataset, test_indices)

    torch.save(train_dataset_dev, output_filepath + '/train_dev.pt')
    torch.save(test_dataset_dev, output_filepath + '/test_dev.pt')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    try:
        main()
    except Exception as e:
        print("Error: ", e)
        input_ = "y"
        input_ = input("Using unsecure connection? [y] ")
        if input_ == 'y' or input_ == '':
            import ssl 
            ssl._create_default_https_context = ssl._create_unverified_context
            main()
        else:
            print("Exiting...")
            exit()
