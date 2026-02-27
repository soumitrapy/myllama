import wandb
from uvf.configs import (
    train_config as TRAIN_CONFIG,
    wandb_config as WANDB_CONFIG,
)
from uvf.utils.config_utils import update_config
from dataclasses import asdict
def setup_wandb(train_config, **kwargs):
    try:
        with open(train_config.wandb_key_path, 'r') as file:
            key = file.read().strip()
            print("Successfully read WandB key.")
    except FileNotFoundError:
        print(f"WandB key file not found at {train_config.wandb_key_path}. Please provide a valid path.")
    except Exception as e:
        print(f"An error occurred during login: {e}")
    wandb.login(key)
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    #run.config.update(fsdp_config, allow_val_change=True)
    return run