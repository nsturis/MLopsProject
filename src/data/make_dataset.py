# -*- coding: utf-8 -*-
from email import generator
from os import ftruncate
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob2
import torch
import numpy as np
from PIL import Image
import PIL
import kornia as K
import albumentations as A
from albumentations.augmentations.geometric import LongestMaxSize
from albumentations.pytorch.transforms import ToTensorV2
import hydra
from hydra.core.config_store import ConfigStore
from src.config import DOGCATConfig

from torch.utils.data import TensorDataset, random_split


def parse_images(file_path, augmentations, data_matrix, labels, c):
    for file in (file_path):
        # Read image
        try:
            img = np.array(Image.open(file).convert("RGB")).astype(np.float32)
            img = augmentations(image = img)['image']
            
            data_matrix.append(img)
            labels.append(c)

        except PIL.UnidentifiedImageError:
            print("Error reading image: " + file)
            continue

    
    return data_matrix, labels


@click.command()
@click.argument('input_folderpath', type=click.Path(exists=True))
@click.argument('output_folderpath', type=click.Path(exists=True))

def main(input_folderpath, output_folderpath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_folderpath = input_folderpath + "/PetImages"

    cat_filepath = glob2.glob(input_folderpath + "/Cat/*.jpg")
    dog_filepath = glob2.glob(input_folderpath + "/Dog/*.jpg")

    # 0 for cats, 1 for dogs
#    labels = torch.concat((torch.zeros(len(cat_filepath)), torch.ones(len(dog_filepath))))

    data_matrix = []
    labels = []

    augmentations = A.Compose([
        LongestMaxSize(200),
        A.PadIfNeeded(200, 200)
        #ToTensorV2()
    ])

    data_matrix, labels = parse_images(cat_filepath, augmentations, data_matrix, labels, 0)
    data_matrix, labels = parse_images(dog_filepath, augmentations, data_matrix, labels, 1)
    # Create a vector of indices to permute the data into train, test and validation sets
    # indices = np.arange(len(data_matrix))
    # np.random.shuffle(indices, generator=np.random.default_rng(42))

    # # Split the data into train, test and validation sets
    # train_size = int(len(data_matrix) * 0.8)
    # test_size = int(len(data_matrix) * 0.1)
    # val_size = int(len(data_matrix) * 0.1)

    # train_indices = indices[:train_size]
    # test_indices = indices[train_size:train_size + test_size]
    # valid_indices = indices[train_size + val_size:]
    data_matrix = np.asarray(data_matrix)
    data_matrix = torch.Tensor(data_matrix).permute(0, 3, 1, 2)
    dataset = TensorDataset(data_matrix, torch.Tensor(labels))

    train_split = 0.8
    test_split = 0.1


    train_size = int(train_split * len(dataset))
    test_size = int(test_split * len(dataset))
    val_size = len(dataset) - train_size - test_size
    
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

    torch.save(train_dataset, output_folderpath + "/train_data.pt")
    torch.save(test_dataset, output_folderpath + "/test_data.pt")
    torch.save(val_dataset, output_folderpath + "/val_data.pt")


    #torch.save(images, output_folderpath + '/animal_train_images.pt')
    #torch.save(labels.long(), output_folderpath + '/animal_test_labels.pt')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
