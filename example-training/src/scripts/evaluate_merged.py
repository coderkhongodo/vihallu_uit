#!/usr/bin/env python3
"""
Script to evaluate merged model on test set.
Uses standard transformers library for inference.
"""

import argparse
import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

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
    Extract label from model output and convert to submission format.
    Model outputs: faithful, intrinsic, extrinsic
    Submission format: no, intrinsic, extrinsic
    """
    if not text:
        return "other"

    # Try to find label in JSON-like format
    m = re.search(r'"?label"?\s*[:=]\s*["\']?\s*(faithful|intrinsic|extrinsic)\b', text, re.IGNORECASE)
    if m:
        label = m.group(1).lower()
        # Convert faithful to no for submission format
        return "no" if label == "faithful" else label

    # Fallback: search for keywords
    m2 = re.search(r'\b(faithful|intrinsic|extrinsic)\b', text, re.IGNORECASE)
    if m2:
        label = m2.group(1).lower()
        # Convert faithful to no for submission format
        return "no" if label == "faithful" else label

    return "other"


def normalize_label(label: str) -> str:
    """
    Normalize label to submission format.
    According to the evaluation criteria:
    - Labels should be: no, intrinsic, extrinsic
    - "faithful" should be converted to "no"
    """
    label = str(label).strip().lower()
    if label == "faithful":
        return "no"
    if label in ["no", "intrinsic", "extrinsic"]:
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
    """Generate prediction for a single sample."""
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
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Parse label
    predicted_label = parse_label(generated_text)
    
    return predicted_label


def evaluate_model(
    model_path: str,
    test_csv: str,
    output_dir: str = "evaluation_results",
    max_new_tokens: int = 20,
    temperature: float = 0.1,
    max_samples: int = None,
):
    """Evaluate model on test set."""
    print(f"\n{'='*70}")
    print(f"EVALUATING MERGED MODEL ON TEST SET")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Test data: {test_csv}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load model and tokenizer
    print("Loading merged model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
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
    label_counts = df['label_normalized'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
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
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70 + "\n")
    
    # Overall accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Per-class metrics
    # According to evaluation criteria: labels are no, intrinsic, extrinsic
    labels = ['no', 'intrinsic', 'extrinsic']
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth,
        predictions,
        labels=labels,
        average=None,
        zero_division=0
    )
    
    print("Per-class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for i, label in enumerate(labels):
        print(f"{label:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Macro and weighted averages
    # Macro-F1 is the primary evaluation metric (for ranking)
    # Formula: Macro-F1 = (1/3) × (F1_no + F1_intrinsic + F1_extrinsic)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, average='weighted', zero_division=0
    )

    print("-" * 70)
    print(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    print(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    print()

    # Verify Macro-F1 calculation
    manual_macro_f1 = sum(f1) / len(labels)
    print(f"Macro-F1 (primary metric): {macro_f1:.4f}")
    print(f"Macro-F1 (manual verification): {manual_macro_f1:.4f}")
    print(f"Accuracy (tie-breaker): {accuracy:.4f}")
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
        "evaluation_criteria": {
            "primary_metric": "macro_f1",
            "tie_breaker": "accuracy",
            "labels": ["no", "intrinsic", "extrinsic"]
        },
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
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
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels
    }
    
    metrics_json = output_path / "metrics.json"
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_json}")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate merged model on test set")
    parser.add_argument("--model-path", type=str, required=True, help="Path to merged model")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_model(
            model_path=args.model_path,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
        )
        
        print("\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

