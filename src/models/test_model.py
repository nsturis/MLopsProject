import os
import hydra
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data.dataloader import AnimalDataModule
from src.config import DOGCATConfig
from model import Classifier
from google.cloud import secretmanager


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DOGCATConfig):
    print("Predicting like a boss")

    PROJECT_ID = "onyx-glider-337908"
    secrets = secretmanager.SecretManagerServiceClient()
    WANDB_KEY = secrets.access_secret_version(
        request={"name": "projects/" + PROJECT_ID + "/secrets/wandb_api_key/versions/1"}
    ).payload.data.decode("utf-8")
    os.environ["WANDB_API_KEY"] = WANDB_KEY

    wandb_logger = WandbLogger(project="MLOps", entity="mlops_flajn", name="Testing")
    wandb_logger.experiment.config.update(cfg)

    # get trained model
    model = Classifier.load_from_checkpoint(
        checkpoint_path=get_original_cwd() + "/models/initial_model.ckpt",
        **{"cfg": cfg}
    )

    # get test dataloader
    data_module = AnimalDataModule(
        batch_size=cfg.model.batch_size,
        data_dir=get_original_cwd() + "/" + cfg.paths.input_filepath,
        image_size=cfg.image.size,
        num_workers=os.cpu_count(),
    )
    test_loader = data_module.test_dataloader()

    # testing
    trainer = Trainer(logger=wandb_logger)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
