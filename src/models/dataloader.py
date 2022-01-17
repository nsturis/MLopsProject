import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import kornia.augmentation as K
from kornia.geometry.transform import resize
from kornia.enhance import normalize
from kornia.utils import image_to_tensor
import numpy as np
from PIL import Image
import json
import torchvision

class DogCatDataset(Dataset):
    def __init__(self, data_split, data_dir, image_size, transform):
        with open(data_dir + "/datasplit.json", "r") as f:
            dict_split = json.loads(f.read())
        self.image_paths = dict_split[data_split]
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_idx = self.image_paths[idx]
        image_org = np.array(Image.open(image_idx).convert("RGB"))
        image = image_to_tensor(image_org).float()
        image = resize(image, (self.image_size, self.image_size), align_corners=True)
        label = 0
        if "Cat" in image_idx:
            label = 1
        
        #torchvision.utils.save_image(image, "test.png", normalize=True)
        if self.transform:
            image = self.transform(image)
        
        #torchvision.utils.save_image(image, "test2.png", normalize=True)
        return image, label
        
class AnimalDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, image_size, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = K.container.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.Normalize(torch.zeros(1), torch.tensor([255])),
            data_keys = ["input"],
            return_transform=False
        )

        self.val_transform = K.container.AugmentationSequential(
            K.Normalize(torch.zeros(1), torch.tensor([255])),
            data_keys = ["input"],
            return_transform=False
        )

    def train_dataloader(self):
        return DataLoader(DogCatDataset("training", data_dir=self.data_dir, image_size=self.image_size, transform=self.train_transform), 
        batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(DogCatDataset("validation", data_dir=self.data_dir, image_size=self.image_size, transform=self.val_transform), 
        batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(DogCatDataset("testing", data_dir=self.data_dir, image_size=self.image_size, transform=self.val_transform), 
        batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    data_module = AnimalDataModule(data_dir='data/processed', image_size=200, batch_size=4, num_workers=4)
    train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

    image, label = next(iter(train_loader))

