#!/usr/bin/env python3
"""
Script to evaluate a model on the test set for hallucination detection.
Calculates accuracy, precision, recall, F1 score, and confusion matrix.
"""

import argparse
import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel


# System prompt for hallucination detection
SYSTEM_PROMPT = """You are a helpful assistant trained to detect hallucinations in model responses.

You will be given three inputs:
1. Context: The reference text containing the factual information.
2. Question: The question to the model.
3. Response: The model's answer.

Your task is to carefully compare the Response against the Context and determine whether the Response is faithful to the Context or hallucinates information.

Label definitions:
- faithful: The response is fully supported by the context. No hallucination.
- intrinsic: The response misinterprets or contradicts the context.
- extrinsic: The response introduces information not present in the context.

Output format: Return only keys:
"label": "<faithful | intrinsic | extrinsic>" """


def parse_label(text: str) -> str:
    """
    Extract label from model output.
    Returns one of faithful|intrinsic|extrinsic or "other".
    """
    if not text:
        return "other"
    
    # Try to find label in JSON-like format
    m = re.search(r'"?label"?\s*[:=]\s*["\']?\s*(faithful|intrinsic|extrinsic)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    
    # Fallback: search for keywords
    m2 = re.search(r'\b(faithful|intrinsic|extrinsic)\b', text, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    
    return "other"


def normalize_label(label: str) -> str:
    """Normalize label to standard format."""
    label = str(label).strip().lower()
    if label == "no":
        return "faithful"
    if label in ["faithful", "intrinsic", "extrinsic"]:
        return label
    return "other"


def generate_prediction(
    model,
    tokenizer,
    context: str,
    prompt: str,
    response: str,
    max_new_tokens: int = 20,
    temperature: float = 0.1,
) -> str:
    """
    Generate prediction for a single sample.
    
    Args:
        model: The model to use for prediction
        tokenizer: The tokenizer
        context: Context text
        prompt: Question/prompt text
        response: Response text to evaluate
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Predicted label (faithful, intrinsic, extrinsic, or other)
    """
    # Construct messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {prompt}\n\nResponse: {response}"
        }
    ]
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Parse label
    predicted_label = parse_label(generated_text)
    
    return predicted_label


def evaluate_model(
    model_path: str,
    test_csv: str,
    output_dir: str = "evaluation_results",
    max_seq_length: int = 2048,
    max_new_tokens: int = 20,
    temperature: float = 0.1,
    batch_size: int = 1,
    max_samples: int = None,
):
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to the model (local or Hugging Face)
        test_csv: Path to test CSV file
        output_dir: Directory to save evaluation results
        max_seq_length: Maximum sequence length
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size (currently only supports 1)
        max_samples: Maximum number of samples to evaluate (None for all)
    """
    print(f"{'='*60}")
    print(f"Model Evaluation on Test Set")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Test data: {test_csv}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    print("✓ Model loaded successfully\n")
    
    # Load test data
    print(f"Loading test data from: {test_csv}")
    df = pd.read_csv(test_csv)
    
    if max_samples:
        df = df.head(max_samples)
        print(f"Limiting evaluation to {max_samples} samples")
    
    print(f"Total test samples: {len(df)}\n")
    
    # Normalize ground truth labels
    df['label_normalized'] = df['label'].apply(normalize_label)
    
    # Print label distribution
    print("Test set label distribution:")
    print(df['label_normalized'].value_counts())
    print()
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    ground_truth = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        context = row.get('context', '')
        prompt = row.get('prompt', '')
        response = row.get('response', '')
        true_label = row['label_normalized']
        
        # Generate prediction
        pred_label = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            context=context,
            prompt=prompt,
            response=response,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        predictions.append(pred_label)
        ground_truth.append(true_label)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60 + "\n")
    
    # Overall accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Per-class metrics
    labels = ['faithful', 'intrinsic', 'extrinsic']
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth,
        predictions,
        labels=labels,
        average=None,
        zero_division=0
    )
    
    print("Per-class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, label in enumerate(labels):
        print(f"{label:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, average='weighted', zero_division=0
    )
    
    print("-" * 60)
    print(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    print(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    print("Confusion Matrix:")
    print(f"{'':>15} " + " ".join([f"{l:>12}" for l in labels]))
    for i, label in enumerate(labels):
        print(f"{label:>15} " + " ".join([f"{cm[i][j]:>12}" for j in range(len(labels))]))
    print()
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_df = df.copy()
    results_df['predicted_label'] = predictions
    results_df['correct'] = [p == g for p, g in zip(predictions, ground_truth)]
    results_csv = output_path / "detailed_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Detailed results saved to: {results_csv}")
    
    # Save metrics summary
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist()
    }
    
    metrics_json = output_path / "metrics.json"
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_json}")
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    
    try:
        evaluate_model(
            model_path=args.model_path,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
        )
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

