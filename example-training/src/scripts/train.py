import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import ConfigHandler
from models import ModelManager
from data import create_datasets
from training import HallucinationTrainer
from utils import setup_environment, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model using unsloth")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--run_number", required=True, help="Run number for this experiment")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    
    config_handler = ConfigHandler(args.config)
    config = config_handler.get_config()
    
    date = datetime.now().strftime("%Y%m%d")
    paths = config_handler.get_formatted_paths(date, args.run_number)
    output_dir = paths["output_dir"]
    run_name = paths["run_name"]
    
    config.setdefault("tracking", {})["run_name"] = run_name
    setup_environment(config)

    formatted_repo_id = config_handler.get_formatted_hf_repo_id(date, args.run_number)
    if formatted_repo_id:
        config.setdefault("huggingface", {})["repo_id"] = formatted_repo_id
    
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model()
    model = model_manager.prepare_peft_model(model)
    tokenizer = model_manager.setup_chat_template(tokenizer)

    train_dataset, eval_dataset = create_datasets(tokenizer, config["data"])

    trainer = HallucinationTrainer(config, model_manager, train_dataset, eval_dataset)
    
    trainer.train(model, tokenizer, output_dir, run_name)

    hf_cfg = config.get("huggingface", {})
    auto_push = bool(hf_cfg.get("push_to_hub", False))
    if not auto_push and hf_cfg.get("repo_id"):
        try:
            repo_id = hf_cfg["repo_id"]
            tag = hf_cfg.get("revision", None)
            model.push_to_hub(repo_id, commit_message="Add fine-tuned checkpoint", revision=tag)
            tokenizer.push_to_hub(repo_id, commit_message="Add tokenizer", revision=tag)
            print(f"Pushed model and tokenizer to Hugging Face: {repo_id}")
        except Exception as e:
            print(f"Warning: Failed to push to Hugging Face Hub: {e}")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
