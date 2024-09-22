import yaml
import os
import shutil
import torch
from torch.distributed import init_process_group



def load_config(cfg_path):
    with open(cfg_path) as file:
        return yaml.safe_load(file)



def initialize_seed(seed):
    # Initialize the random seed for both CPU and GPU.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def build_env(init_cfg_path, cfg_name, exp_path, ckp_path, result_path):
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    cfg_path = os.path.join(exp_path, cfg_name)
    if init_cfg_path != cfg_path :
        shutil.copyfile(init_cfg_path,cfg_path)



def initialize_process_group(cfg, rank):
    """Initialize the process group for distributed training."""
    init_process_group(
        backend=cfg['env_config']['dist_config']['dist_backend'],
        init_method=cfg['env_config']['dist_config']['dist_url'],
        world_size=cfg['env_config']['dist_config']['world_size'] * cfg['env_config']['num_gpus'],
        rank=rank
    )



def log_model_info(model, exp_path):
    """Log model information and create necessary directories."""
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("Generator Parameters :", num_params)
    os.makedirs(exp_path, exist_ok=True)
    print("checkpoints directory :", exp_path)



def LoadPretrainedPath(path, name, cfg):
    if path != None:
        return path
    return cfg['pretrained_config'][name]
