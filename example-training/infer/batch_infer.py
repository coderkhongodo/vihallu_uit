import asyncio
import argparse
import csv
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import httpx
from tqdm.auto import tqdm

from internalllm import LLMServerProvider


SYSTEM_PROMPT = (
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
    'Output format: Return only keys:\n"label": "<faithful | intrinsic | extrinsic>"'
)


def parse_label(text: str) -> str:
    """Extract label from model output. Returns one of faithful|intrinsic|extrinsic or "other"."""
    if not text:
        return "other"
    m = re.search(r'"?label"?\s*[:=]\s*["\']?\s*(faithful|intrinsic|extrinsic)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r'\b(faithful|intrinsic|extrinsic)\b', text, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    return "other"


async def one_call(
    session: httpx.AsyncClient,
    provider: LLMServerProvider,
    messages: List[Dict[str, Any]],
    settings: Dict[str, Any],
    model: str,
) -> Tuple[str, str]:
    output, reasoning = await provider.create_chat_completion(
        session=session,
        messages=messages,
        settings=settings,
        model=model,
        thinking_mode=False,
    )
    return output or "", reasoning or ""


@dataclass
class Sample:
    id: str
    context: str
    prompt: str
    response: str


def read_samples(input_csv: str) -> List[Sample]:
    rows: List[Sample] = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Sample(
                    id=str(r.get("id", "")).strip(),
                    context=r.get("context", ""),
                    prompt=r.get("prompt", ""),
                    response=r.get("response", ""),
                )
            )
    return rows


def choose_final_label(counts: Counter) -> str:
    vote_labels = {k: counts.get(k, 0) for k in ("faithful", "intrinsic", "extrinsic")}
    if sum(vote_labels.values()) == 0:
        return "faithful" 
    order = {"intrinsic": 2, "extrinsic": 1, "faithful": 0}
    best: Optional[str] = None
    best_count = -1
    for lbl, c in vote_labels.items():
        if c > best_count or (c == best_count and order[lbl] > order.get(best or "faithful", -1)):
            best = lbl
            best_count = c
    return best or "faithful"


def map_to_submit_label(raw_label: str) -> str:
    if raw_label == "faithful":
        return "no"
    if raw_label in ("intrinsic", "extrinsic"):
        return raw_label
    return "no"


async def infer_for_sample(
    session: httpx.AsyncClient,
    provider: LLMServerProvider,
    sample: Sample,
    model: str,
    repeat: int,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    vary_seed: bool,
    quiet: bool,
) -> Tuple[Counter, str, str]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context: {sample.context}\n\nQuestion: {sample.prompt}\n\nResponse: {sample.response}",
        },
    ]

    base_settings = provider.get_provider_default_settings().as_dict()
    base_settings["max_tokens"] = max_tokens
    base_settings["temperature"] = temperature

    counts: Counter = Counter()
    errors = 0

    async def call_once(i: int) -> Optional[str]:
        local_settings = dict(base_settings)  # avoid mutation across coroutines
        if vary_seed:
            local_settings["seed"] = i + 1
        output, _ = await one_call(session, provider, messages, local_settings, model)
        return parse_label(output)

    total_runs = max(1, repeat)
    batch_size = max(1, min(concurrency, total_runs))
    for start in range(0, total_runs, batch_size):
        end = min(start + batch_size, total_runs)
        tasks = [asyncio.create_task(call_once(i)) for i in range(start, end)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, res in enumerate(results, 1):
            if isinstance(res, Exception):
                errors += 1
                if not quiet:
                    print(f"{sample.id} | run {start+idx} failed: {res}")
                continue
            lbl = res or "other"
            counts[lbl] += 1
            if not quiet:
                print(f"{sample.id} | run {start+idx}: {lbl}")

    return counts, str(errors), str(sum(counts.values()))


async def main_async(args) -> None:
    input_csv = args.input_csv
    output_dir = args.output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    samples = read_samples(input_csv)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    provider = LLMServerProvider(f"http://{args.host}:{args.port}")

    counts_path = os.path.join(output_dir, "counts.csv")
    submit_path = os.path.join(output_dir, "submit.csv")
    zip_path = os.path.join(output_dir, "submit.zip")

    async with httpx.AsyncClient(timeout=60.0) as session:
        with open(counts_path, "w", encoding="utf-8", newline="") as f_counts, \
             open(submit_path, "w", encoding="utf-8", newline="") as f_submit:
            counts_writer = csv.writer(f_counts)
            counts_writer.writerow([
                "id", "faithful", "intrinsic", "extrinsic", "other", "total", "errors", "predicted_raw", "predict_label"
            ])
            submit_writer = csv.writer(f_submit)
            submit_writer.writerow(["id", "predict_label"])

            for idx, s in enumerate(tqdm(samples, total=len(samples), desc="Samples"), 1):
                counts, errors, total = await infer_for_sample(
                    session=session,
                    provider=provider,
                    sample=s,
                    model=args.model,
                    repeat=args.repeat,
                    concurrency=args.concurrency,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    vary_seed=args.vary_seed,
                    quiet=args.quiet,
                )
                raw = choose_final_label(counts)
                submit_lbl = map_to_submit_label(raw)

                counts_writer.writerow([
                    s.id,
                    counts.get("faithful", 0),
                    counts.get("intrinsic", 0),
                    counts.get("extrinsic", 0),
                    counts.get("other", 0),
                    total,
                    errors,
                    raw,
                    submit_lbl,
                ])
                submit_writer.writerow([s.id, submit_lbl])

    try:
        import zipfile

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(submit_path, arcname="submit.csv")
    except Exception as e:
        print(f"Failed to create zip: {e}")

    if not args.quiet:
        print(f"\nWrote:\n- {counts_path}\n- {submit_path}\n- {zip_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference for hallucination labeling and submission generation")
    p.add_argument("--input-csv", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vihallu-public-test.csv"), help="Path to input CSV with id,context,prompt,response,predict_label")
    p.add_argument("--output-dir", type=str, default=os.getcwd(), help="Directory to write counts.csv, submit.csv, and submit.zip")
    p.add_argument("--host", type=str, default="127.0.0.1", help="LLM server host")
    p.add_argument("--port", type=int, default=8080, help="LLM server port")
    p.add_argument("--model", type=str, default="hallu", help="Model name")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=10, help="Max tokens")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--repeat", type=int, default=64, help="Votes per sample")
    p.add_argument("--concurrency", type=int, default=64, help="Concurrent requests per sample")
    p.add_argument("--vary-seed", action="store_true", help="Vary seed across repeats")
    p.add_argument("--quiet", action="store_true", help="Less logging")
    p.add_argument("--limit", type=int, default=0, help="Optional: only process first N samples")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
