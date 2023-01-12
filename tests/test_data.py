import torch
from src.data.make_dataset import main
import os


def test_data_size(): 

    input_filepath = 'data/raw/'
    output_path = 'data/processed'
    main([input_filepath, output_path])
    assert os.path.exists(output_path + '/train.pt') == True 
    assert os.path.exists(output_path + '/test.pt') == True
    assert os.path.exists(output_path + '/test_dev.pt') == True
    assert os.path.exists(output_path + '/train_dev.pt') == True





if __name__ == '__main__':
    test_data_size()