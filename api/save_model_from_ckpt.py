import hydra
import torch
import pytorch_lightning as pl
from src.config import DOGCATConfig, register_configs
from hydra.utils import get_original_cwd
from src.models.model import Classifier

register_configs()

@hydra.main(config_path="../src/conf", config_name="config")
def main(cfg: DOGCATConfig):

    model = Classifier.load_from_checkpoint(get_original_cwd() + "/api/initial_model.ckpt", cfg=cfg)
    m = torch.jit.script(model)
    torch.jit.save(m, get_original_cwd() + "/api/model.pt")
    

if __name__ == "__main__":
    main()