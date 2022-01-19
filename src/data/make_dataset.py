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
from src.config import DOGCATConfig
import random
import json


def parse_images(file_path, valid_files):
    for file in (file_path):
        # Read image
        try:
            _ = np.array(Image.open(file).convert("RGB")).astype(np.float32)
            valid_files.append(file)

        except PIL.UnidentifiedImageError:
            print("Error reading image: " + file)
            continue

    
    return valid_files


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

    valid_files = []
    valid_files = parse_images(cat_filepath, valid_files)
    valid_files = parse_images(dog_filepath, valid_files)
    random.Random(42).shuffle(valid_files)
    train_split = 0.8
    test_split = 0.1
    train_size = int(train_split * len(valid_files))
    test_size = int(test_split * len(valid_files))
    val_size = len(valid_files) - train_size - test_size
    train_indices = valid_files[:train_size]
    test_indices = valid_files[train_size:train_size + test_size]
    valid_indices = valid_files[train_size + val_size:]

    dict_split = {"training": train_indices, "validation": valid_indices, "testing": test_indices}

    with open(output_folderpath + '/datasplit.json', 'w') as f:
        json.dump(dict_split, f)
        f.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
