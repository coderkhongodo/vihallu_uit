import pandas as pd
import torch
from datasets import Dataset as HFDataset
from typing import Dict, Any, Tuple

from .dataset import HallucinationDataset


def create_datasets(
    tokenizer, 
    data_config: Dict[str, Any]
) -> Tuple[HFDataset, HFDataset]:
    if "csv_path" not in data_config:
        raise ValueError("'csv_path' must be specified for hallucination detection task")
    
    df = pd.read_csv(data_config["csv_path"]) 
    for col in ["id", "context", "prompt", "response", "label"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    data = df.to_dict("records")

    dataset = HallucinationDataset(
        tokenizer=tokenizer,
        data=data,
        max_length=data_config.get("max_length", 2048),
        system_content=data_config.get("system_content"),
    )

    train_split = data_config.get("train_split", 0.8)
    val_split = data_config.get("val_split", 0.2)
    if abs(train_split + val_split - 1.0) > 1e-6:
        raise ValueError("train_split + val_split must equal 1.0 when test_split is omitted")
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    eval_size = total_size - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(data_config.get("random_seed", 42))
    )

    def to_hf(subset):
        return HFDataset.from_dict({
            "input_ids": [item["input_ids"].tolist() for item in subset],
            "attention_mask": [item["attention_mask"].tolist() for item in subset],
            "labels": [item["labels"].tolist() for item in subset],
        })

    return to_hf(train_dataset), to_hf(eval_dataset)
