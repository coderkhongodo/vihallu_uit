from typing import Dict, Any, List
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

from .callbacks import (
    PerplexityCallback,
    HubPushLoggerCallback,
)


class HallucinationTrainer:
    def __init__(
            self, 
            config: Dict[str, Any], 
            model_manager, 
            train_dataset, 
            eval_dataset
        ):
        self.config = config
        self.model_manager = model_manager
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
    def create_training_arguments(
            self,
            output_dir: str,
            run_name: str | None = None,
    ) -> TrainingArguments:
        training_config = self.config["training"]
        hf_cfg = self.config["huggingface"]
        
        report_to = self.config.get("tracking", {}).get("report_to", [])
        if isinstance(report_to, str):
            report_to = [report_to]

        return TrainingArguments(
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            warmup_ratio=training_config["warmup_ratio"],
            num_train_epochs=training_config["num_train_epochs"],
            learning_rate=float(training_config["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=training_config["logging_steps"],
            optim=training_config["optim"],
            weight_decay=training_config["weight_decay"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            save_steps=training_config["save_steps"],
            seed=training_config["seed"],
            output_dir=output_dir,
            report_to=report_to,
            save_strategy=training_config["save_strategy"],
            save_total_limit=training_config["save_total_limit"],
            do_eval=training_config["do_eval"],
            eval_strategy=training_config["eval_strategy"],
            eval_steps=training_config["eval_steps"],
            load_best_model_at_end=True,
            metric_for_best_model=self.config["early_stopping"]["metric"],
            push_to_hub=bool(hf_cfg["push_to_hub"]),
            hub_model_id=hf_cfg["repo_id"],
            hub_token=hf_cfg["token"],
            hub_strategy="checkpoint",
            run_name=run_name
        )
    
    def create_sft_config(self) -> SFTConfig:
        return SFTConfig(
            max_length=self.config["model"]["max_seq_length"],
            dataset_num_proc=self.config["training"]["dataset_num_proc"],
            packing=False,
        )

    def create_callbacks(self) -> List:
        callbacks = [
            PerplexityCallback(),
            HubPushLoggerCallback(),
        ]
        
        if "early_stopping" in self.config:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=self.config["early_stopping"]["patience"],
            )
            callbacks.append(early_stopping_callback)
        
        return callbacks
    
    def create_trainer(
            self,
            model,
            tokenizer,
            output_dir: str,
            run_name: str | None = None,
    ) -> SFTTrainer:
        training_args = self.create_training_arguments(output_dir, run_name)
        sft_config = self.create_sft_config()
        callbacks = self.create_callbacks()
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            callbacks=callbacks,
            args=training_args,
            **sft_config.__dict__
        )
        
        return trainer
    
    def train(
            self,
            model,
            tokenizer,
            output_dir: str,
            run_name: str,
    ) -> None:
        trainer = self.create_trainer(model, tokenizer, output_dir, run_name)

        print("Dataset loaded successfully:")
        print(f"  Training samples: {len(self.train_dataset):,}")
        print(f"  Evaluation samples: {len(self.eval_dataset):,}")

        trainer.train()
