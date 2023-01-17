import torch 
from omegaconf import OmegaConf
import yaml
from src.models_lightning.model import MyAwesomeModel
from src.models.train_model import main

def test_train():

    with open('src/models/conf/unit_test.yaml', 'r') as config:
        config = yaml.safe_load(config)
         
    config = {'experiment': config}
    cfg = OmegaConf.create(config)
    running_loss = main(cfg)
    assert running_loss != 0

if __name__ =='__main__':
    test_train()