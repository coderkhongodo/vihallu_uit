#!/usr/bin/env python3
"""
Script to generate predictions for private test set and create submit.csv file.
Output format: id, predict_label (values: "no", "extrinsic", "intrinsic")
"""

import argparse
import sys
import os
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Extract label from model output."""
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


def convert_to_submit_format(label: str) -> str:
    """
    Convert model output label to submission format.
    faithful -> no
    intrinsic -> intrinsic
    extrinsic -> extrinsic
    other -> no (default to no hallucination)
    """
    if label == "faithful":
        return "no"
    elif label == "intrinsic":
        return "intrinsic"
    elif label == "extrinsic":
        return "extrinsic"
    else:
        # Default to "no" for uncertain cases
        return "no"


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
    
    # Convert to submission format
    submit_label = convert_to_submit_format(predicted_label)
    
    return submit_label


def predict_and_create_submission(
    model_path: str,
    test_csv: str,
    output_file: str = "submit.csv",
    max_new_tokens: int = 20,
    temperature: float = 0.1,
    max_samples: int = None,
):
    """Generate predictions and create submission file."""
    print(f"\n{'='*70}")
    print(f"GENERATING PREDICTIONS FOR SUBMISSION")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Test data: {test_csv}")
    print(f"Output file: {output_file}")
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
        print(f"Limiting prediction to {max_samples} samples")
    
    print(f"Total test samples: {len(df)}\n")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    ids = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        sample_id = row['id']
        context = row.get('context', '')
        prompt = row.get('prompt', '')
        response = row.get('response', '')
        
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
        
        ids.append(sample_id)
        predictions.append(pred_label)
    
    # Create submission dataframe
    submit_df = pd.DataFrame({
        'id': ids,
        'predict_label': predictions
    })
    
    # Print label distribution
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70 + "\n")
    
    label_counts = submit_df['predict_label'].value_counts()
    print("Predicted label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(submit_df)*100:.1f}%)")
    print()
    
    # Save submission file
    submit_df.to_csv(output_file, index=False)
    print(f"✓ Submission file saved to: {output_file}")
    
    # Verify format
    print("\nVerifying submission format...")
    verify_df = pd.read_csv(output_file)
    
    # Check columns
    if list(verify_df.columns) == ['id', 'predict_label']:
        print("✓ Columns are correct: ['id', 'predict_label']")
    else:
        print(f"✗ Warning: Columns are {list(verify_df.columns)}, expected ['id', 'predict_label']")
    
    # Check values
    valid_labels = {'no', 'intrinsic', 'extrinsic'}
    unique_labels = set(verify_df['predict_label'].unique())
    if unique_labels.issubset(valid_labels):
        print(f"✓ All labels are valid: {unique_labels}")
    else:
        invalid = unique_labels - valid_labels
        print(f"✗ Warning: Found invalid labels: {invalid}")
    
    # Check number of rows
    print(f"✓ Number of predictions: {len(verify_df)}")
    
    print(f"\n{'='*70}")
    print("SUBMISSION FILE CREATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nFile: {output_file}")
    print(f"Format: id, predict_label")
    print(f"Valid labels: no, intrinsic, extrinsic")
    print(f"Total predictions: {len(verify_df)}")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for submission")
    parser.add_argument("--model-path", type=str, required=True, help="Path to merged model")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to private test CSV")
    parser.add_argument("--output-file", type=str, default="submit.csv", help="Output submission file")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to predict (for testing)")
    
    args = parser.parse_args()
    
    try:
        predict_and_create_submission(
            model_path=args.model_path,
            test_csv=args.test_csv,
            output_file=args.output_file,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
        )
        
        print("\n✓ Prediction completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

