import torch 
from omegaconf import DictConfig, OmegaConf
import yaml
from src.models.model import MyAwesomeModel
from src.models.train_model import main

def test_train():

    with open('src/models/conf/experiment/unit_test.yaml', 'r') as config:
        config = yaml.safe_load(config)
         
    config = {'experiment': config}
    cfg = OmegaConf.create(config)
    main(cfg)
    assert True

if __name__ =='__main__':
    test_train()