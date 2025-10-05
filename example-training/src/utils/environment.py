import os
from typing import Dict, Any


def setup_environment(config: Dict[str, Any]) -> None:
    tracking = config.get("tracking", {})
    hf_cfg = config.get("huggingface", {})

    env_vars = {
        "WANDB_API_KEY": tracking.get("wandb_api_key", ""),
        "WANDB_PROJECT": tracking.get("wandb_project", ""),
        "WANDB_ENTITY": tracking.get("wandb_entity", ""),
        "WANDB_MODE": tracking.get("wandb_mode", "online"),
        "WANDB_NAME": tracking.get("run_name", ""),
        "HF_TOKEN": hf_cfg.get("token", ""),
        "HUGGING_FACE_HUB_TOKEN": hf_cfg.get("token", ""),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        
    print("Environment variables configured for WANDB and Hugging Face")
