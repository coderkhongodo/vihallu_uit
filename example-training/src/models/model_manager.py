from typing import Dict, Any, Tuple, Optional
from unsloth import FastLanguageModel, get_chat_template
from transformers import AutoTokenizer, AutoModelForCausalLM

from .model_utils import ModelUtils


class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config["model"]
        self.lora_config = config["lora"]
        self.utils = ModelUtils(config)
    
    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load base model and tokenizer using unsloth"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config["name_or_path"],
            max_seq_length=self.model_config["max_seq_length"],
            load_in_4bit=self.model_config.get("load_in_4bit", False),
            dtype=self.model_config.get("dtype", None),
        )
        
        return model, tokenizer
    
    def prepare_peft_model(
            self,
            model: AutoModelForCausalLM
    ) -> AutoModelForCausalLM:
        """Prepare model for PEFT training with LoRA"""
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config["r"],
            target_modules=self.lora_config["target_modules"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config.get("lora_dropout", 0.0),
            bias=self.lora_config.get("bias", "none"),
            use_gradient_checkpointing=self.lora_config.get("use_gradient_checkpointing", True),
            random_state=self.lora_config.get("random_state", 3407),
            use_rslora=self.lora_config.get("use_rslora", False),
            loftq_config=self.lora_config.get("loftq_config", None),
        )
        
        return model
    
    def setup_chat_template(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
        """Setup chat template for instruction following"""
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.model_config.get("chat_template", "qwen3-instruct"),
            mapping={
                "role": "role", 
                "content": "message", 
                "user": "user", 
                "assistant": "assistant", 
                "system": "system"
            },
            map_eos_token=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """Find the latest checkpoint in output directory"""
        return self.utils.find_latest_checkpoint(output_dir)
    
    def load_from_checkpoint(
            self, 
            checkpoint_path: str
        ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model from checkpoint"""
        return self.utils.load_from_checkpoint(checkpoint_path, self.model_config)
    
    def save_model(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        output_path: str
    ) -> None:
        """Save model and tokenizer"""
        self.utils.save_model(model, tokenizer, output_path)
