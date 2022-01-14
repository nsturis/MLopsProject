import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
import kornia.augmentation as K
import numpy as np

# class AnimalDataset(Dataset):
#     def __init__(self, data_path, data_split, image_size, transform):
#         self.images = torch.load(data_path + "/" + data_split + "_images.pt")
#         self.labels = torch.load(data_path + "/" + data_split + "_labels.pt")
#         self.height = image_size
#         self.width = image_size
#         self.transform = transform

#     def __len(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]

#        # if self.transform:
#         #    image = self.transform(image)

#         return image, label
        
class AnimalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, image_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_workers = num_workers

        self.train_transform = nn.Sequential(
            #K.RandomHorizontalFlip(0.5),
            K.Normalize(torch.zeros(3), torch.ones(3))
        )

        self.val_transform = nn.Sequential(
            K.Normalize(torch.zeros(3), torch.ones(3))
        )


    def train_dataloader(self):
        dataset = torch.load(self.data_dir + '/train_data.pt')
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        #dataloader = self.train_transform(dataloader)
        # Impelemt transforms in train_step
        return dataloader
    
    def val_dataloader(self):
        dataset = torch.load(self.data_dir + '/val_data.pt')
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        #dataloader = self.val_transform(dataloader)
        return dataloader
    
    def test_dataloader(self):
        dataset = torch.load(self.data_dir + '/test_data.pt')
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        #dataloader = self.val_transform(dataloader)
        return dataloader

if __name__ == "__main__":
    data_module = AnimalDataModule(batch_size=4, data_dir='data/processed', image_size=200, num_workers=4)
    train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels.shape)
        break