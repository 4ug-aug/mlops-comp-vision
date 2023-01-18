import logging
import os

from model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np
import hydra
from hydra.utils import get_original_cwd
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from utils import count_files

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

    # change working dir
    os.chdir(get_original_cwd()) 

    # set experiment
    cfg = cfg.experiment

    # define model
    model = MyAwesomeModel(classes=cfg.hyperparameters.n_classes,
                           lr = cfg.hyperparameters.lr,
                           weight_decay = cfg.hyperparameters.weight_decay,
                           batch_size = cfg.hyperparameters.batch_size,
                           optimizer = cfg.hyperparameters.optimizer,
                           dataset_path = cfg.hyperparameters.dataset_path,
                           num_workers=cfg.hyperparameters.num_workers, 
                           pretrained=cfg.hyperparameters.pretrained)

    # define trainer
    logger = WandbLogger(project=cfg.hyperparameters.wandb_project)
    logger.log_hyperparams(cfg.hyperparameters)
    trainer = Trainer(max_epochs=cfg.hyperparameters.max_epochs, 
                      logger=logger, 
                      log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())
    trainer.test(model, dataloaders=model.test_dataloader())
    
    new_path = count_files(f"{get_original_cwd()}/models/trained_models")
    new_path = f"{get_original_cwd()}/models/trained_models/model_checkpoint_{new_path}.pth"

    print("Saving model as model_checkpoint.pth")
    print("Path: {}".format(new_path))
    
    torch.save(model.cnn, new_path)


    
    # plot the training loss
    #plt.plot(training_loss, label='Training loss')
    #plt.legend(frameon=False)

    #new_path = count_files(get_original_cwd() + "/reports/figures")

    # Save plot
    #plt.savefig(f"{get_original_cwd()}/reports/figures/training_loss_{new_path}.png")

    # Generate unique name for model
    #new_path = count_files(get_original_cwd() + "/models/trained_models")
    #new_path = f"{get_original_cwd()}/models/trained_models/model_checkpoint_{new_path}.pth"

    #print("Saving model as model_checkpoint.pth")
    #print("Path: {}".format(new_path))
    #torch.save(model, new_path)


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