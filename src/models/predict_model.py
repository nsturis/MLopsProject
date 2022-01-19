import os
import hydra
import wandb
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data.dataloader import AnimalDataModule
from src.config import DOGCATConfig
from model import Classifier
import json
from PIL import Image
from google.cloud import secretmanager

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DOGCATConfig):
    print("Predicting like a boss")

    PROJECT_ID = 'onyx-glider-337908'
    secrets = secretmanager.SecretManagerServiceClient()
    WANDB_KEY = secrets.access_secret_version(request={"name": "projects/"+PROJECT_ID+"/secrets/wandb_api_key/versions/1"}).payload.data.decode("utf-8")
    os.environ['WANDB_API_KEY'] = WANDB_KEY

    wandb_logger = WandbLogger(project='MLOps', entity='mlops_flajn', name = 'Prediction')
    wandb_logger.experiment.config.update(cfg)

    # get trained model
    model = Classifier.load_from_checkpoint(checkpoint_path = get_original_cwd() + '/models/initial_model.ckpt', **{"cfg":cfg})

    # get test dataloader
    data_dir = get_original_cwd() + "/" + cfg.paths.input_filepath
    data_module = AnimalDataModule(batch_size=cfg.model.batch_size, data_dir=data_dir, image_size=cfg.image.size, num_workers=os.cpu_count())
    test_loader = data_module.test_dataloader()

    # predict
    trainer = Trainer(logger=wandb_logger, limit_predict_batches=1)
    predictions = trainer.predict(model, dataloaders=test_loader)[0]

    # log predictions in wandb as a table
    images, labels = next(iter(test_loader))
    labels_dict = {0:'Dog',1:'Cat'}
    with open(data_dir + "/datasplit.json", "r") as f:
        dict_split = json.loads(f.read())
    raw_images_paths = dict_split['testing'][:len(predictions)]
    columns=["id", "label", "raw_image","processed_image", "prediction"]
    data = []
    for i in range(len(images)):
        raw_img = Image.open(raw_images_paths[i]).convert("RGB")
        file_name = raw_images_paths[i].split("PetImages/")[1]
        data.append([
            file_name,
            labels_dict[labels[i].item()],
            wandb.Image(raw_img),
            wandb.Image(images[i]),
            labels_dict[predictions[i].item()]
        ])
    wandb_logger.log_table(key='first_batch_prediction', columns=columns, data=data)


if __name__ == "__main__":
    main()