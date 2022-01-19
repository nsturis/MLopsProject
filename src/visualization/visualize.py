import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import hydra
from hydra.utils import get_original_cwd

import torch
from torchvision.utils import make_grid, save_image
from kornia.utils import image_to_tensor

from src.data.dataloader import AnimalDataModule, DogCatDataset
from src.config import DOGCATConfig

@hydra.main(config_path="../conf", config_name="config")
def visualize_processed_data(cfg: DOGCATConfig):
    data_dir = get_original_cwd() + "/" + cfg.paths.input_filepath
    data_module = AnimalDataModule(batch_size=cfg.model.batch_size, data_dir=data_dir, image_size=200, num_workers=os.cpu_count())
    train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

    num_pictures = 8

    with open(data_dir + "/datasplit.json", "r") as f:
        dict_split = json.loads(f.read())

    images, _ = next(iter(train_loader))
    processed_images = images[:num_pictures]

    raw_images_paths = dict_split['training'][:num_pictures]
    rows = 2

    plt.figure(figsize=(80,20))
    index=0

    for anImage in raw_images_paths:
        img = Image.open(anImage).convert("RGB")
        plt.subplot(rows,8,index+1)
        plt.axis('off')
        plt.imshow(img)
        index+=1

    for anImage in processed_images:
        plt.subplot(rows,8,index+1)
        plt.axis('off')
        plt.imshow(anImage.permute(1, 2, 0))
        index+=1
    
    print('Generating batch_train_visualization')
    plt.savefig(get_original_cwd() + "/" + 'reports/figures/batch_train_visualization.jpg')


if __name__ == "__main__":
    visualize_processed_data()

    