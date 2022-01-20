from logging.config import valid_ident
import os
import pytest
import numpy as np
import torch

from omegaconf import OmegaConf

from tests import _PATH_DATA

from src.data.dataloader import AnimalDataModule


@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}"),
    reason="Data folder doesn't exist yet. Please execute MAKE DATA.",
)
def test_processed_files_created():
    """Check hthat all data/processed/ files are created"""
    assert os.path.exists(f"{_PATH_DATA}/processed/test_data.pt")
    assert os.path.exists(f"{_PATH_DATA}/processed/train_data.pt")
    assert os.path.exists(f"{_PATH_DATA}/processed/val_data.pt")
    assert os.path.exists(f"{_PATH_DATA}/processed/datasplit.json")


@pytest.fixture
def get_config():
    """Initialisation : get config file"""
    cfg = OmegaConf.load("src/conf/config.yaml")
    return cfg


@pytest.fixture
def get_dataloaders():
    """Initialisation : get dataloaders"""
    cfg = OmegaConf.load("src/conf/config.yaml")
    data_module = AnimalDataModule(
        batch_size=cfg.model.batch_size,
        data_dir=f"{_PATH_DATA}/processed/",
        image_size=cfg.image.size,
        num_workers=os.cpu_count(),
    )
    train_loader, val_loader, test_loader = (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader(),
    )
    return train_loader, val_loader, test_loader


@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}"),
    reason="Data folder doesn't exist yet. Please execute MAKE DATA.",
)
def test_image_shape(get_dataloaders, get_config):
    cfg = get_config
    train_loader, val_loader, test_loader = get_dataloaders

    def loader_check_shape(aDataLoader):
        dataiter = iter(aDataLoader)
        images, _ = next(dataiter)
        assert len(images.shape) == 4, "Images are not a 4D tensor"
        assert images.shape[2] == cfg.image.size, "Image size is incorrect"
        assert images.shape[3] == cfg.image.size, "Image size is incorrect"

    loader_check_shape(train_loader)
    loader_check_shape(test_loader)
    loader_check_shape(val_loader)


@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}"),
    reason="Data folder doesn't exist yet. Please execute MAKE DATA.",
)
def test_contain_cat_and_dog(get_dataloaders):
    """Check that each dataset contains at least one dog and one cat"""
    train_loader, val_loader, test_loader = get_dataloaders

    def loader_check_labels(aDataLoader):
        found_0, found_1 = False, False
        for _, labels in iter(aDataLoader):
            set_labels = set(list(labels.numpy().flatten()))
            if 0 in set_labels:
                found_0 = True
            if 1 in set_labels:
                found_1 = True
            if found_0 and found_1:
                return True
        return False

    assert loader_check_labels(train_loader), "Train dataset doesn't contain all labels"
    assert loader_check_labels(test_loader), "Test dataset doesn't contain all labels"
    assert loader_check_labels(
        val_loader
    ), "Validation dataset doesn't contain all labels"
