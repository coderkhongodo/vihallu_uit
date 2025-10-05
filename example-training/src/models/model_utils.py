import os
import re
from typing import Dict, Any, Tuple, Optional
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelUtils:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """Find the latest checkpoint in output directory"""
        if not os.path.exists(output_dir):
            return None
            
        checkpoints = [d for d in os.listdir(output_dir) if re.match(r"checkpoint-\d+", d)]
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        return os.path.join(output_dir, latest)
    
    def load_from_checkpoint(
        self, 
        checkpoint_path: str, 
        model_config: Dict[str, Any]
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model from checkpoint"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=model_config["max_seq_length"],
            load_in_4bit=model_config.get("load_in_4bit", False),
            dtype=model_config.get("dtype", None),
        )
        
        return model, tokenizer
    
    def save_model(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        output_path: str
    ) -> None:
        """Save model and tokenizer to specified path"""
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Model saved to: {output_path}")
