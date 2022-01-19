import torch
import os
import hydra
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer
from src.data.dataloader import AnimalDataModule
from src.config import DOGCATConfig
from model import Classifier

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DOGCATConfig):
    print("Evaluating until hitting the ceiling")

    # get trained model
    model = Classifier.load_from_checkpoint(checkpoint_path = get_original_cwd() + '/models/initial_model.ckpt', **{"cfg":cfg})

    # get test dataloader
    data_module = AnimalDataModule(batch_size=cfg.model.batch_size, data_dir=get_original_cwd() + "/" + cfg.paths.input_filepath, image_size=cfg.image.size, num_workers=os.cpu_count())
    test_loader = data_module.test_dataloader()

    # predict
    trainer = Trainer()
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()