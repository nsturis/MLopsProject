# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob2
import torch
import numpy as np
from PIL import Image
import kornia as K

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
    for file in (cat_filepath):
        # Read image
        try:
            img = np.array(Image.open(file).convert("RGB")).astype(np.double)
            img = torch.Tensor(img)
            img = K.geometry.transform.warp_perspective(torch.permute(torch.Tensor(img)[None],(3,0,1,2)), torch.eye(3)[None], (224, 224))
            # save img to file
            #print(img.shape)
        except:
            print("Error reading image: " + file)
            continue
        # Double-precision floating
        I = np.double(img)

        data_matrix.append(I)
        labels.append(0)

    for file in dog_filepath:
        # Read image
        try:
            img = np.array(Image.open(file).convert("RGB")).astype(np.double)
        except:
            print("Error reading image: " + file)
            continue
        # Double-precision floating
        I = np.double(img)
        data_matrix.append(I)
        labels.append(1)

    data_matrix = np.array(data_matrix)
    images = torch.from_numpy(data_matrix)

    torch.save(images, output_folderpath + '/animal_images.pt')
    torch.save(labels.long(), output_folderpath + '/animal_labels.pt')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
