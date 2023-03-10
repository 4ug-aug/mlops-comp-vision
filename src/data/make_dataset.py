# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
# import numpy as np
# import timm
# import os
import pandas as pd
# from torchvision import datasets, transforms
import torch
from dotenv import find_dotenv, load_dotenv
# from PIL import Image
# from tqdm import tqdm
from utils import PIL_to_tensor


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to download data
    from torchvision and save it in ../processed.

    Args:
        input_filepath (str): Path to the raw data
        output_filepath (str): Path to the processed data

    Returns:
        None

    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((224, 224)),
        ]
    )
    """

    # for us. run this once to run locally
    # train_dataset = datasets.CIFAR10(input_filepath, download=True, train=True, transform=transform)
    # test_dataset = datasets.CIFAR10(input_filepath, download=True, train=False, transform=transform)

    data = pd.read_csv(input_filepath + "/BUTTERFLIES.csv")

    # train
    trainset = data[data["data set"] == "train"]

    train_imgs = PIL_to_tensor(input_filepath, trainset["filepaths"])

    train_labels = torch.Tensor(trainset["class index"].values)

    trainset = torch.utils.data.TensorDataset(train_imgs.float(), train_labels.long())

    torch.save(trainset, output_filepath + "/train.pt")

    # used for unit testing
    n_classes = 20
    dev_size_per_class = 5
    # indexes to make sure each class is represented
    indexes = []
    occurrences = [0 for _ in range(n_classes)]
    for i, idx in enumerate(train_labels.long()):
        if occurrences[idx] == dev_size_per_class:
            continue
        indexes.append(i)
        occurrences[idx] += 1

    trainset_dev = torch.utils.data.TensorDataset(
        train_imgs[indexes].float(), train_labels[indexes].long()
    )

    torch.save(trainset_dev, output_filepath + "/train_dev.pt")

    # test

    testset = data[data["data set"] == "test"]

    test_imgs = PIL_to_tensor(input_filepath, testset["filepaths"])

    test_labels = torch.Tensor(testset["class index"].values)

    testset = torch.utils.data.TensorDataset(test_imgs.float(), test_labels.long())

    torch.save(testset, output_filepath + "/test.pt")

    # validation

    valset = data[data["data set"] == "valid"]

    val_imgs = PIL_to_tensor(input_filepath, valset["filepaths"])

    val_labels = torch.Tensor(valset["class index"].values)

    valset = torch.utils.data.TensorDataset(val_imgs.float(), val_labels.long())

    torch.save(valset, output_filepath + "/val.pt")

    # Save the data
    # torch.save(train_dataset, output_filepath + '/train.pt')
    # torch.save(test_dataset, output_filepath + '/test.pt')

    # make dev set
    # train_indices = torch.randperm(len(train_dataset))[:5000]
    # train_dataset_dev = torch.utils.data.Subset(train_dataset, train_indices)

    # test_indices = torch.randperm(len(test_dataset))[:1000]
    # test_dataset_dev = torch.utils.data.Subset(test_dataset, test_indices)

    # torch.save(train_dataset_dev, output_filepath + '/train_dev.pt')
    # torch.save(test_dataset_dev, output_filepath + '/test_dev.pt')


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

    """
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
    """
