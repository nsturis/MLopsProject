import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
import kornia.augmentation as K
import numpy as np

class AnimalDataset(Dataset):
    def __init__(self, data_path, image_size, transform):
        self.images = torch.load(data_path + "/animal_images.pt")
        self.labels = torch.load(data_path + "/animal_labels.pt")
        self.height = image_size
        self.width = image_size
        self.transform = transform

    def __len(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transforms(image)

        return img, label
        

class AnimalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, image_size):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.image_size = image_size

        self.train_transform = nn.Sequential(
            K.RandomHorizontalFlip(0.5),
            K.Normalize(torch.zeros(3), torch.ones(3))
        )

        self.val_transform = nn.Sequential(
            K.Normalize(torch.zeros(3), torch.ones(3))
        )

    def get_dataloaders(self):
        dataset = AnimalDataset(data_dir = self.data_dir, image_size = self.image_size, transform = self.train_transform)
        train_split = 0.8
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size = self.batch_size)
        return train_dataloader, val_dataloader

if __name__ == "__main__":
    print("Y")