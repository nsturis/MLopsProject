import os
import hydra
from hydra.utils import get_original_cwd
from src.data.dataloader import AnimalDataModule
from src.config import DOGCATConfig
from google.cloud import secretmanager
import torch
import wandb
import torchmetrics
import numpy as np


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DOGCATConfig):
    print("Testing up to heaven")

    PROJECT_ID = "onyx-glider-337908"
    secrets = secretmanager.SecretManagerServiceClient()
    WANDB_KEY = secrets.access_secret_version(
        request={"name": "projects/" + PROJECT_ID + "/secrets/wandb_api_key/versions/1"}
    ).payload.data.decode("utf-8")
    os.environ["WANDB_API_KEY"] = WANDB_KEY

    wandb.init(config=cfg, project="MLOps", entity="mlops_flajn", name="Testing")

    # get trained model
    model = torch.jit.load(get_original_cwd() + '/models/model_bigboy.pt')

    # get test dataloader
    data_module = AnimalDataModule(
        batch_size=cfg.model.batch_size,
        data_dir=get_original_cwd() + "/" + cfg.paths.input_filepath,
        image_size=cfg.image.size,
        num_workers=os.cpu_count(),
    )
    test_loader = data_module.test_dataloader()

    # testing
    all_acc, all_loss = [],[]
    i = 0
    for X, y in iter(test_loader):
        print(f"Step {i+1}")
        logits = model(X)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        test_acc = torchmetrics.Accuracy()
        acc = test_acc(preds, y)
        all_acc.append(acc)
        all_loss.append(loss)
        i+=1

    all_acc, all_loss = torch.tensor(np.array(all_acc)), torch.tensor(np.array(all_loss))

    print("Test_loss = ", all_loss.mean())
    print("Test_acc = ", all_acc.mean())
    wandb.log({"test_accuracy": all_acc.mean()})
    wandb.log({"test_loss": all_loss.mean()})

if __name__ == "__main__":
    main()
