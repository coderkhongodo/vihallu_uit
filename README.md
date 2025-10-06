# ViHallu UIT - Vietnamese Hallucination Detection

Dự án phát hiện hallucination (ảo giác) trong các mô hình ngôn ngữ lớn (LLM) cho tiếng Việt.

## 📊 Kết quả

**Best Model: Merged LoRA Model**
- **Macro-F1**: 0.8977 (89.77%)
- **Accuracy**: 0.8971 (89.71%)
- Base Model: Qwen/Qwen2.5-7B-Instruct
- LoRA Adapter: QuangDuy/lora-hallu

### Per-class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **no** (faithful) | 92.61% | 95.09% | 93.83% | 224 |
| **intrinsic** | 91.74% | 86.12% | 88.84% | 245 |
| **extrinsic** | 85.00% | 88.31% | 86.62% | 231 |

## 🏗️ Cấu trúc Project

```
vihallu_uit/
├── example-training/
│   ├── data/                          # Dữ liệu train/val/test
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── src/
│       └── scripts/
│           ├── train.py               # Script training
│           ├── merge_lora.py          # Merge LoRA adapter với base model
│           ├── evaluate.py            # Evaluate model
│           ├── evaluate_merged.py     # Evaluate merged model (BEST)
│           ├── multi_agent_debate_v6_improved.py  # Multi-agent debate (V6 Keyword)
│           └── predict_submit.py      # Tạo submission file
├── evaluation_results/                # Kết quả evaluation (Macro-F1: 0.8977)
│   ├── metrics.json
│   └── detailed_results.csv
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/coderkhongodo/vihallu_uit.git
cd vihallu_uit
```

### 2. Cài đặt dependencies

```bash
pip install -r example-training/src/requirements.txt
```

### 3. Tải models

**Base Model:**
```bash
# Model sẽ tự động tải từ HuggingFace khi chạy script
# Qwen/Qwen2.5-7B-Instruct
```

**LoRA Adapter:**
```bash
# Adapter sẽ tự động tải từ HuggingFace
# QuangDuy/lora-hallu
```

## 📝 Sử dụng

### 1. Merge LoRA với Base Model

```bash
cd example-training/src/scripts
python merge_lora.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter QuangDuy/lora-hallu \
    --output-dir ../../../merged_model
```

### 2. Evaluate Merged Model

```bash
python evaluate_merged.py \
    --model-path ../../../merged_model \
    --test-csv ../../data/test.csv \
    --output-dir ../../../evaluation_results
```

### 3. Predict cho Submission

```bash
python predict_submit.py \
    --model-path ../../../merged_model \
    --test-file ../../../vihallu-private-test.csv \
    --output-file ../../../submit.csv
```

### 4. Multi-Agent Debate (Experimental)

```bash
python multi_agent_debate_v6_improved.py \
    --model-path ../../../merged_model \
    --test-csv ../../data/test.csv \
    --output-dir ../../../debate_results \
    --max-samples 100
```

## 📊 Định nghĩa Labels

- **no (faithful)**: Response được hỗ trợ hoàn toàn bởi Context, không có hallucination
- **intrinsic**: Response MÂU THUẪN / ĐẢO NGƯỢC thông tin trong Context
- **extrinsic**: Response THÊM thông tin mới mà Context không đề cập

## 🔬 Phương pháp

### Single-Agent Approach (BEST - 89.77% Macro-F1)

Sử dụng merged model (Qwen2.5-7B-Instruct + LoRA adapter) để phân loại trực tiếp.

**Ưu điểm:**
- Đơn giản, hiệu quả
- Performance cao nhất
- Inference nhanh

### Multi-Agent Debate (Experimental - ~74% Macro-F1)

Sử dụng 3 agents tranh luận:
- Agent 1 (Accuser): Tìm hallucination
- Agent 2 (Defender): Kiểm tra lại
- Agent 3 (Judge): Phán quyết cuối cùng

**Ưu điểm:**
- Giải thích rõ ràng
- Có thể cải thiện với model lớn hơn

**Nhược điểm:**
- Performance thấp hơn single-agent với 7B model
- Inference chậm (3x)

## 📈 Kết quả Chi tiết

Xem file `evaluation_results/metrics.json` và `evaluation_results/detailed_results.csv` để biết kết quả chi tiết.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License

## 👥 Tác giả

- GitHub: [@coderkhongodo](https://github.com/coderkhongodo)
- Project: [vihallu_uit](https://github.com/coderkhongodo/vihallu_uit)

## 🙏 Acknowledgments

- Base Model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- LoRA Adapter: [QuangDuy/lora-hallu](https://huggingface.co/QuangDuy/lora-hallu)
- Dataset: ViHallu UIT Competition

