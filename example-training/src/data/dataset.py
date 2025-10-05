import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional


class HallucinationDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data: List[Dict[str, Any]],
            max_length: int = 2048,
            system_content: Optional[str] = None
        ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.system_content = system_content or self._get_default_system_content()
        self.processed_data = self._process_data()

    def _get_default_system_content(self) -> str:
        return (
            "You are a helpful assistant trained to detect hallucinations in model responses.\n\n"
            "You will be given three inputs:\n"
            "1. Context: The reference text containing the factual information.\n"
            "2. Question: The question to the model.\n"
            "3. Response: The model's answer.\n\n"
            "Your task is to carefully compare the Response against the Context and determine whether the Response is faithful to the Context or hallucinates information.\n\n"
            "Label definitions:\n"
            "- faithful: The response is fully supported by the context. No hallucination.\n"
            "- intrinsic: The response misinterprets or contradicts the context.\n"
            "- extrinsic: The response introduces information not present in the context.\n\n"
            "Output format: Return only keys:\n"
            "\"label\": \"<faithful | intrinsic | extrinsic>\""
        )

    def _process_data(self) -> List[Dict[str, torch.Tensor]]:
        processed = []

        for item in self.data:
            context = item.get("context", "")
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            label = str(item.get("label", "faithful")).strip().lower()

            if label == "no":
                label = "faithful"
            if label not in ["faithful", "intrinsic", "extrinsic"]:
                label = "faithful"

            messages = [
                {
                    "role": "system", 
                    "content": self.system_content
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {prompt}\n\nResponse: {response}",
                },
                {
                    "role": "assistant",
                    "content": f'"label": "{label}"',
                },
            ]

            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            full_tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]

            labels = self._create_masked_labels(full_text, input_ids)

            processed.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

        return processed

    def _create_masked_labels(
            self,
            full_text: str,
            input_ids: List[int]
        ) -> List[int]:
        labels = input_ids.copy()

        assistant_start_pattern = "<|im_start|>assistant\n"
        last_assistant_pos = full_text.rfind(assistant_start_pattern)

        if last_assistant_pos != -1:
            assistant_content_start = last_assistant_pos + len(assistant_start_pattern)

            assistant_end_pattern = "<|im_end|>"
            assistant_end_pos = full_text.find(assistant_end_pattern, assistant_content_start)

            if assistant_end_pos != -1:
                assistant_end_with_token = assistant_end_pos + len(assistant_end_pattern)
                assistant_response_text = full_text[assistant_content_start:assistant_end_with_token]

                text_before_assistant = full_text[:assistant_content_start]
                tokens_before = self.tokenizer(text_before_assistant, add_special_tokens=False)["input_ids"]

                assistant_tokens = self.tokenizer(assistant_response_text, add_special_tokens=False)["input_ids"]

                for i in range(len(labels)):
                    if i < len(tokens_before) or i >= len(tokens_before) + len(assistant_tokens):
                        labels[i] = -100
            else:
                labels = [-100] * len(labels)
        else:
            labels = [-100] * len(labels)

        return labels

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]
