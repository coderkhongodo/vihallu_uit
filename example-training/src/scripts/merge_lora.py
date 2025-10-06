#!/usr/bin/env python3
"""
Script to merge LoRA adapter with base model.
Downloads the LoRA adapter from Hugging Face and merges it with the base model.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
import torch


def merge_lora_adapter(
    lora_model_id: str,
    base_model_id: str = None,
    output_dir: str = "merged_model",
    max_seq_length: int = 2048,
    dtype = None,
    load_in_4bit: bool = False,
):
    """
    Merge LoRA adapter with base model.
    
    Args:
        lora_model_id: Hugging Face model ID for the LoRA adapter
        base_model_id: Hugging Face model ID for the base model (if None, will be inferred from adapter config)
        output_dir: Directory to save the merged model
        max_seq_length: Maximum sequence length
        dtype: Data type for model weights
        load_in_4bit: Whether to load in 4-bit quantization
    """
    print(f"{'='*60}")
    print(f"Merging LoRA Adapter with Base Model")
    print(f"{'='*60}")
    print(f"LoRA adapter: {lora_model_id}")
    if base_model_id:
        print(f"Base model: {base_model_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load the LoRA model
    print("Loading LoRA adapter and base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    print("✓ Model and tokenizer loaded successfully\n")
    
    # Enable inference mode for faster inference
    print("Preparing model for inference...")
    FastLanguageModel.for_inference(model)
    print("✓ Model prepared for inference\n")
    
    # Merge LoRA weights with base model
    print("Merging LoRA weights with base model...")
    print("This may take a few minutes...")
    
    # Save the merged model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving merged model to: {output_path}")
    
    # Save using unsloth's save_pretrained_merged method
    # This automatically merges the LoRA weights and saves the full model
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",  # Save as full 16-bit model
    )
    
    print(f"✓ Merged model saved successfully to: {output_path}")
    print(f"\n{'='*60}")
    print(f"Merge Complete!")
    print(f"{'='*60}")
    print(f"\nYou can now use the merged model from: {output_path}")
    print(f"The model can be loaded with standard transformers or unsloth.")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model"
    )
    parser.add_argument(
        "--lora-model-id",
        type=str,
        default="QuangDuy/lora-hallu",
        help="Hugging Face model ID for the LoRA adapter (default: QuangDuy/lora-hallu)"
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default=None,
        help="Hugging Face model ID for the base model (optional, will be inferred from adapter)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="merged_model",
        help="Directory to save the merged model (default: merged_model)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    try:
        merge_lora_adapter(
            lora_model_id=args.lora_model_id,
            base_model_id=args.base_model_id,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
        )
        print("\n✓ Merge completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

