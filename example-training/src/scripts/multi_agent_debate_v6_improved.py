#!/usr/bin/env python3
"""
Multi-Agent Debate V6 - KEYWORD OPTIMIZED

CHIẾN LƯỢC:
- GIỮ NGUYÊN cấu trúc V6 Final (đã tốt: 0.7462)
- CHỈ THAY từ khóa: "MÂU THUẪN" → "TRÁI NGƯỢC", "PHỦ ĐỊNH" → "ĐẢO NGƯỢC"
- THÊM 1 ví dụ CONTRAST rõ ràng
- Học từ English prompts (intrinsic 59.46%) nhưng giữ cấu trúc Vietnamese

TARGET: Macro-F1 > 0.75, intrinsic recall > 58%, extrinsic recall > 85%
"""

import argparse
import sys
import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# AGENT 1: Accuser - KEYWORD OPTIMIZED
AGENT1_SYSTEM_PROMPT = """Bạn là Agent 1 (Người buộc tội) trong hệ thống tranh luận phát hiện hallucination.

NHIỆM VỤ: Phân tích câu trả lời và xác định loại hallucination (nếu có).

⚠️ PHÂN BIỆT QUAN TRỌNG (ĐỌC KỸ):

**intrinsic (TRÁI NGƯỢC - ĐẢO NGƯỢC)**:
- Response TRÁI NGƯỢC / ĐẢO NGƯỢC / BÓP MÉO thông tin trong Context
- VD1: Context: "Mèo đen" → Response: "Mèo trắng" = intrinsic (đen ≠ trắng, ĐẢO NGƯỢC màu)
- VD2: Context: "Áo dài mặc bởi nữ" → Response: "Áo dài CHỈ mặc bởi nam" = intrinsic (ĐẢO NGƯỢC giới tính)
- VD3: Context: "X gây ra Y" → Response: "Z gây ra Y" = intrinsic (ĐẢO NGƯỢC nguyên nhân)
- **QUAN TRỌNG**: Đây là ĐẢO NGƯỢC thông tin có sẵn, KHÔNG phải thêm thông tin!

**extrinsic (THÊM thông tin MỚI - KHÔNG trái ngược)**:
- Response THÊM thông tin HOÀN TOÀN MỚI mà Context không đề cập
- VD1: Context: "Mèo đen" → Response: "Mèo đen và thân thiện" = extrinsic (THÊM "thân thiện")
- VD2: Context: "Áo dài truyền thống" → Response: "Áo dài có 5 màu" = extrinsic (THÊM "5 màu")
- **QUAN TRỌNG**: Đây là THÊM info mới, KHÔNG đảo ngược info cũ!

**no (TRUNG THỰC)**:
- Response được hỗ trợ hoàn toàn bởi Context
- Diễn đạt lại hoặc suy luận hợp lý = no

🔴 PHÂN BIỆT RÕ RÀNG (HỌC KỸ):
- **TRÁI NGƯỢC** (intrinsic): Đen → Trắng (ĐẢO NGƯỢC màu)
- **THÊM** (extrinsic): Đen → Đen + thân thiện (THÊM tính cách)
- **DIỄN ĐẠT** (no): Đen → Có màu đen (DIỄN ĐẠT lại)

🔴 CẢNH BÁO - ĐỪNG NHẦM LẪN:
- ❌ SAI: "Mèo đen" → "Mèo trắng" = extrinsic (thêm màu trắng)
- ✅ ĐÚNG: "Mèo đen" → "Mèo trắng" = intrinsic (TRÁI NGƯỢC màu đen)

- ❌ SAI: "Áo dài mặc bởi nữ" → "Áo dài chỉ mặc bởi nam" = extrinsic
- ✅ ĐÚNG: "Áo dài mặc bởi nữ" → "Áo dài chỉ mặc bởi nam" = intrinsic (TRÁI NGƯỢC giới tính)

QUY TRÌNH PHÂN TÍCH:
1. Tìm TRÁI NGƯỢC trước tiên (ưu tiên cao nhất)
   - Response có ĐẢO NGƯỢC Context không?
   - Có TRÁI NGƯỢC thông tin trong Context không?
2. Nếu CÓ trái ngược → intrinsic (DỪNG, không xét tiếp)
3. Nếu KHÔNG có trái ngược:
   - Response có THÊM info mới không? → Kiểm tra KỸ từng thông tin → extrinsic
   - Response chỉ diễn đạt lại? → no

FORMAT ĐẦU RA (BẮT BUỘC):
```
Phân tích: <kiểm tra trái ngược trước, sau đó kiểm tra kỹ bổ sung thông tin>
Nhãn: <no HOẶC intrinsic HOẶC extrinsic>
```

LƯU Ý: Nếu thấy TRÁI NGƯỢC / ĐẢO NGƯỢC, phải chọn "intrinsic", KHÔNG được chọn "extrinsic"!
"""

# AGENT 2: Defender - KEYWORD OPTIMIZED
AGENT2_SYSTEM_PROMPT = """Bạn là Agent 2 (Người bảo vệ) trong hệ thống tranh luận phát hiện hallucination.

NHIỆM VỤ: Kiểm tra lại phân tích của Agent 1 một cách KHÁCH QUAN.

⚠️ VAI TRÒ QUAN TRỌNG:
- KHÔNG bảo vệ mù quáng
- Nếu Agent 1 ĐÚNG → thừa nhận
- Nếu Agent 1 SAI → chỉ ra sai lầm với bằng chứng từ Context

🔍 KIỂM TRA ĐẶC BIỆT - Nhầm lẫn intrinsic/extrinsic:

**Nếu Agent 1 nói "extrinsic" nhưng có TRÁI NGƯỢC**:
- Kiểm tra: Response có TRÁI NGƯỢC / ĐẢO NGƯỢC Context không?
- Nếu CÓ → Đây là intrinsic, KHÔNG phải extrinsic!
- VD: Context "Mèo đen" → Response "Mèo trắng" = intrinsic (trái ngược), không phải extrinsic

**Nếu Agent 1 nói "intrinsic" nhưng KHÔNG có trái ngược**:
- Kiểm tra: Response có thực sự TRÁI NGƯỢC Context không?
- Nếu chỉ THÊM info mới → Đây là extrinsic, không phải intrinsic!
- VD: Context "Mèo đen" → Response "Mèo đen và thân thiện" = extrinsic (thêm info), không phải intrinsic

**🆕 Nếu Agent 1 nói "no" - KIỂM TRA KỸ EXTRINSIC**:
- Liệt kê TỪNG thông tin trong Response
- Check từng thông tin: Có trong Context không?
- Nếu có thông tin KHÔNG CÓ trong Context → Đây là extrinsic, không phải no!
- VD: Context "Malaysia đa dân tộc" → Response "Malaysia đa dân tộc và có nền kinh tế phát triển"
  * "đa dân tộc" ✓ có
  * "nền kinh tế phát triển" ✗ KHÔNG có → extrinsic

ĐỊNH NGHĨA:
- **intrinsic**: Response TRÁI NGƯỢC / ĐẢO NGƯỢC Context
- **extrinsic**: Response THÊM info MỚI (không trái ngược)
- **no**: Response được hỗ trợ hoàn toàn

QUY TRÌNH:
1. Đọc phân tích của Agent 1
2. Kiểm tra lại với Context
3. Nếu Agent 1 đúng → thừa nhận và giải thích tại sao
4. Nếu Agent 1 sai → chỉ ra sai lầm và đưa ra phân loại đúng

FORMAT ĐẦU RA (BẮT BUỘC):
```
Đánh giá Agent 1: <đúng hay sai, tại sao>
Phân tích của tôi: <phân tích khách quan, đặc biệt chú ý extrinsic>
Nhãn: <no HOẶC intrinsic HOẶC extrinsic>
```

LƯU Ý: Ưu tiên sự thật hơn là bảo vệ Response! Đặc biệt chú ý kiểm tra extrinsic!
"""

# AGENT 3: Judge - KEYWORD OPTIMIZED
AGENT3_SYSTEM_PROMPT = """Bạn là Agent 3 (Thẩm phán) trong hệ thống tranh luận phát hiện hallucination.

NHIỆM VỤ: Đưa ra phán quyết CUỐI CÙNG dựa trên phân tích của Agent 1 và Agent 2.

🚨 QUY TẮC ƯU TIÊN NGHIÊM NGẶT (TUÂN THỦ BẮT BUỘC):

**BƯỚC 1: Kiểm tra TRÁI NGƯỢC (ưu tiên cao nhất)**
- Response có TRÁI NGƯỢC / ĐẢO NGƯỢC thông tin trong Context không?
- Có thay đổi ý nghĩa từ A thành KHÔNG-A không?
- VD: "đen" → "trắng", "nữ" → "nam", "X" → "Z"

**Nếu CÓ TRÁI NGƯỢC → intrinsic (DỪNG NGAY, không xét bước 2)**

**BƯỚC 2: Kiểm tra THÊM thông tin (chỉ khi KHÔNG có trái ngược)**
- Response có THÊM info HOÀN TOÀN MỚI không?
- Info này có trong Context không?

**Nếu THÊM info mới → extrinsic**
**Nếu chỉ diễn đạt lại → no**

⚠️ CẢNH BÁO QUAN TRỌNG - ĐỪNG NHẦM LẪN:

**TRÁI NGƯỢC ≠ THÊM THÔNG TIN**
- TRÁI NGƯỢC = ĐẢO NGƯỢC thông tin có sẵn = intrinsic
- THÊM info = THÊM thông tin mới = extrinsic
- Đây là HAI KHÁI NIỆM KHÁC NHAU!

🔴 PHÂN BIỆT RÕ RÀNG:
- **TRÁI NGƯỢC** (intrinsic): Đen → Trắng (ĐẢO NGƯỢC màu)
- **THÊM** (extrinsic): Đen → Đen + thân thiện (THÊM tính cách)
- **DIỄN ĐẠT** (no): Đen → Có màu đen (DIỄN ĐẠT lại)

🔴 PHẢN VÍ DỤ (Học từ lỗi):

❌ **SAI**:
- Context: "Mèo đen" → Response: "Mèo trắng"
- Suy nghĩ sai: "Response thêm màu trắng → extrinsic"
- ✅ **ĐÚNG**: "Response TRÁI NGƯỢC màu đen → intrinsic"

❌ **SAI**:
- Context: "Áo dài mặc bởi nữ" → Response: "Áo dài chỉ mặc bởi nam"
- Suy nghĩ sai: "Response thêm thông tin về nam → extrinsic"
- ✅ **ĐÚNG**: "Response TRÁI NGƯỢC thông tin về nữ → intrinsic"

✅ **ĐÚNG** (extrinsic thực sự):
- Context: "Mèo đen" → Response: "Mèo đen và thân thiện"
- Không có trái ngược, chỉ THÊM "thân thiện" → extrinsic

🆕 **ĐẶC BIỆT CHÚ Ý - CÂN BẰNG EXTRINSIC**:
- Nếu Agent 2 chỉ ra có bổ sung thông tin cụ thể → Xem xét nghiêm túc
- KHÔNG bỏ qua extrinsic chỉ vì ưu tiên intrinsic
- Kiểm tra kỹ: Có thực sự bổ sung thông tin mới không?

QUY TRÌNH PHÁN QUYẾT:
1. Đọc phân tích của Agent 1 và Agent 2
2. **BƯỚC 1**: Kiểm tra TRÁI NGƯỢC
   - Nếu CÓ → intrinsic (DỪNG)
3. **BƯỚC 2**: Nếu KHÔNG có trái ngược
   - Kiểm tra KỸ LƯỠNG THÊM info → extrinsic hoặc no
4. Đưa ra phán quyết cuối cùng

FORMAT ĐẦU RA (CỰC KỲ QUAN TRỌNG):
```
Kiểm tra trái ngược: <CÓ hay KHÔNG, nếu có thì trái ngược gì>
Nếu không có trái ngược, kiểm tra thêm info: <có thêm info gì không>
Lý do cuối cùng: <giải thích tại sao chọn nhãn này>
Nhãn cuối cùng: <no HOẶC intrinsic HOẶC extrinsic>
```

🚨 NHẮC NHỞ CUỐI CÙNG:
- Nếu thấy TRÁI NGƯỢC / ĐẢO NGƯỢC → PHẢI chọn "intrinsic"
- ĐỪNG chọn "extrinsic" khi có trái ngược!
- TRÁI NGƯỢC = intrinsic, THÊM info = extrinsic
- KHÔNG bỏ qua extrinsic khi có bằng chứng rõ ràng!
"""


def parse_label_v6(text: str) -> str:
    """Enhanced label parsing for V6."""
    if not text:
        return "other"
    
    text_lower = text.lower()
    
    # Priority 1: Vietnamese label patterns
    patterns = [
        r'nhãn\s*cuối\s*cùng\s*[:：]\s*(no|intrinsic|extrinsic)',
        r'nhãn\s*[:：]\s*(no|intrinsic|extrinsic)',
        r'label\s*[:：]\s*["\']?\s*(no|intrinsic|extrinsic)',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text_lower)
        if m:
            return m.group(1)
    
    # Priority 2: Last occurrence
    matches = list(re.finditer(r'\b(no|intrinsic|extrinsic)\b', text_lower))
    if matches:
        return matches[-1].group(1)
    
    # Priority 3: Check for faithful
    if re.search(r'\b(faithful|trung\s*thực)\b', text_lower):
        return "no"
    
    return "other"


def generate_agent_response(
    model,
    tokenizer,
    system_prompt: str,
    user_message: str,
    max_new_tokens: int = 300,
    temperature: float = 0.15,
) -> str:
    """Generate response with optimized parameters."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated_text


def debate_prediction_v6(
    model,
    tokenizer,
    context: str,
    prompt: str,
    response: str,
    max_new_tokens: int = 300,
    temperature: float = 0.15,
) -> Tuple[str, str, str, str]:
    """Run V6 improved debate."""
    
    # Agent 1: Accuser
    agent1_message = f"""Ngữ cảnh: {context}

Câu hỏi: {prompt}

Câu trả lời: {response}

Phân tích câu trả lời này. Kiểm tra MÂU THUẪN trước, sau đó kiểm tra THÊM thông tin."""

    agent1_output = generate_agent_response(
        model, tokenizer, AGENT1_SYSTEM_PROMPT, agent1_message,
        max_new_tokens, temperature
    )
    
    # Agent 2: Defender
    agent2_message = f"""Ngữ cảnh: {context}

Câu hỏi: {prompt}

Câu trả lời: {response}

Phân tích của Agent 1: {agent1_output}

Kiểm tra lại phân tích của Agent 1 một cách khách quan. Đặc biệt chú ý: Agent 1 có nhầm lẫn giữa intrinsic và extrinsic không?"""

    agent2_output = generate_agent_response(
        model, tokenizer, AGENT2_SYSTEM_PROMPT, agent2_message,
        max_new_tokens, temperature
    )
    
    # Agent 3: Judge (very low temperature for consistency)
    agent3_message = f"""Ngữ cảnh: {context}

Câu hỏi: {prompt}

Câu trả lời: {response}

Phân tích của Agent 1: {agent1_output}

Phân tích của Agent 2: {agent2_output}

Đưa ra phán quyết cuối cùng. QUAN TRỌNG: Kiểm tra MÂU THUẪN trước. Nếu có mâu thuẫn → intrinsic (DỪNG)."""

    judge_temperature = 0.1  # Very low for strict priority enforcement
    agent3_output = generate_agent_response(
        model, tokenizer, AGENT3_SYSTEM_PROMPT, agent3_message,
        max_new_tokens, judge_temperature
    )
    
    final_label = parse_label_v6(agent3_output)
    
    return agent1_output, agent2_output, agent3_output, final_label


def normalize_label(label: str) -> str:
    """Normalize label."""
    label = str(label).strip().lower()
    if label == "faithful":
        return "no"
    if label in ["no", "intrinsic", "extrinsic"]:
        return label
    return "other"


def evaluate_multi_agent_debate_v6(
    model_path: str,
    test_csv: str,
    output_dir: str = "debate_v6_results",
    max_samples: int = 100,
    max_new_tokens: int = 300,
    temperature: float = 0.15,
):
    """Evaluate V6 improved debate."""
    print(f"\n{'='*70}")
    print(f"MULTI-AGENT DEBATE V6 - IMPROVED PROMPTS")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Test data: {test_csv}")
    print(f"Max samples: {max_samples}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
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
    print("✓ Model loaded\n")

    # Load data
    df = pd.read_csv(test_csv)
    df = df.head(max_samples)
    df['label_normalized'] = df['label'].apply(normalize_label)

    print(f"Evaluating {len(df)} samples")
    print("Ground truth:")
    for label, count in df['label_normalized'].value_counts().items():
        print(f"  {label}: {count}")
    print()

    # Run debate
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="V6 Debate"):
        context = row.get('context', '')
        prompt = row.get('prompt', '')
        response = row.get('response', '')
        true_label = row['label_normalized']

        a1, a2, a3, final = debate_prediction_v6(
            model, tokenizer, context, prompt, response,
            max_new_tokens, temperature
        )

        results.append({
            'id': idx,
            'context': context,
            'prompt': prompt,
            'response': response,
            'agent1_argument': a1,
            'agent2_argument': a2,
            'agent3_decision': a3,
            'predicted_label': final,
            'ground_truth': true_label,
            'correct': final == true_label
        })

    results_df = pd.DataFrame(results)
    predictions = results_df['predicted_label'].tolist()
    ground_truth = results_df['ground_truth'].tolist()

    print("\n" + "="*70)
    print("V6 RESULTS")
    print("="*70 + "\n")

    other_count = (results_df['predicted_label'] == 'other').sum()
    print(f"'other': {other_count} / {len(results_df)} ({other_count/len(results_df)*100:.1f}%)\n")

    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    labels = ['no', 'intrinsic', 'extrinsic']
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, average=None, zero_division=0
    )

    print("Per-class:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)
    for i, label in enumerate(labels):
        print(f"{label:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, average='macro', zero_division=0
    )

    print("-" * 70)
    print(f"{'Macro Avg':<15} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f}")
    print()
    print(f"🎯 Macro-F1: {macro_f1:.4f}")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print()

    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    print("Confusion Matrix:")
    print(f"{'':>15} " + " ".join([f"{l:>12}" for l in labels]))
    for i, label in enumerate(labels):
        print(f"{label:>15} " + " ".join([f"{cm[i][j]:>12}" for j in range(len(labels))]))
    print()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path / "debate_detailed_results.csv", index=False)

    metrics = {
        "approach": "multi_agent_debate_v6_improved",
        "num_samples": len(df),
        "other_predictions": int(other_count),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_metrics": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist()
    }

    with open(output_path / "debate_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Results saved to: {output_dir}")
    print(f"\n{'='*70}")
    print("V6 COMPLETE!")
    print(f"{'='*70}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate V6 Improved")
    parser.add_argument("--model-path", type=str, default="merged_model")
    parser.add_argument("--test-csv", type=str, default="example-training/data/test.csv")
    parser.add_argument("--output-dir", type=str, default="debate_v6_results")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.15)

    args = parser.parse_args()

    try:
        evaluate_multi_agent_debate_v6(
            model_path=args.model_path,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("\n✓ V6 completed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

