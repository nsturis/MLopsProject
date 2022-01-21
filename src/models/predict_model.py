import os
import hydra
import wandb
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer
from src.data.dataloader import AnimalDataModule
from src.config import DOGCATConfig
import torch
import json
from PIL import Image
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
    wandb.init(config=cfg, project="MLOps", entity="mlops_flajn", name="Prediction")

    # get trained model
    model = torch.jit.load(get_original_cwd() + '/models/model_bigboy.pt')

    # get test dataloader
    data_dir = get_original_cwd() + "/" + cfg.paths.input_filepath
    data_module = AnimalDataModule(
        batch_size=cfg.model.batch_size,
        data_dir=data_dir,
        image_size=cfg.image.size,
        num_workers=os.cpu_count(),
    )
    test_loader = data_module.test_dataloader()

    # predict
    images, labels = next(iter(test_loader))
    predictions = torch.argmax(model(images),dim=1)


    # log predictions in wandb as a table
    labels_dict = {0: "Dog", 1: "Cat"}
    with open(data_dir + "/datasplit.json", "r") as f:
        dict_split = json.loads(f.read())
    raw_images_paths = dict_split["testing"][: len(predictions)]
    columns = ["id", "label", "raw_image", "processed_image", "prediction"]
    table = wandb.Table(columns=columns)
    for i in range(len(images)):
        raw_img = Image.open(raw_images_paths[i]).convert("RGB")
        file_name = raw_images_paths[i].split("PetImages/")[1]
        table.add_data(file_name,
                labels_dict[labels[i].item()],
                wandb.Image(raw_img),
                wandb.Image(images[i]),
                labels_dict[predictions[i].item()])
    
    
    wandb.log({"first_batch_prediction":table})


if __name__ == "__main__":
    main()
