# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from model import Classifier
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.config import DOGCATConfig, register_configs
import hydra
from hydra.utils import get_original_cwd
from src.data.dataloader import AnimalDataModule
from google.cloud import secretmanager
import torch
import wandb
register_configs()


# data/processed reports/figures models
@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DOGCATConfig):
    print("Training day and night")

    print(cfg)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_original_cwd()+'/onyx-glider-337908-7017e9498613.json'
    PROJECT_ID = "onyx-glider-337908"
    secrets = secretmanager.SecretManagerServiceClient()
    WANDB_KEY = secrets.access_secret_version(
        request={"name": "projects/" + PROJECT_ID + "/secrets/wandb_api_key/versions/1"}
    ).payload.data.decode("utf-8")
    os.environ["WANDB_API_KEY"] = WANDB_KEY

    wandb_logger = WandbLogger(
        project="MLOps", entity="mlops_flajn", name="100 epoch speedrun"
    )
    wandb_logger.experiment.config.update(cfg)

    data_module = AnimalDataModule(
        batch_size=cfg.model.batch_size,
        data_dir=get_original_cwd() + "/" + cfg.paths.input_filepath,
        image_size=cfg.image.size,
        num_workers=12,
    )
    train_loader, val_loader = (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
    )

    model = Classifier(cfg)

    trainer = Trainer(gpus=-1, max_epochs=100, log_every_n_steps=100, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
    m = torch.jit.script(model)
    torch.jit.save(m, get_original_cwd() + "/src/models/model_100_epoch_speedrun.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
