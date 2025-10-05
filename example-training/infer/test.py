import asyncio
import argparse
import re
from collections import Counter
from typing import List, Dict, Any, Tuple

import httpx

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
    "Output format: Return only keys:\n"
    "\"label\": \"<faithful | intrinsic | extrinsic>\""
)


def parse_label(text: str) -> str:
    """Extract label from model output. Returns one of faithful|intrinsic|extrinsic or "other".

    Tries JSON-like patterns first, then falls back to searching keywords.
    """
    if not text:
        return "other"
    m = re.search(r'"?label"?\s*[:=]\s*["\']?\s*(faithful|intrinsic|extrinsic)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r'\b(faithful|intrinsic|extrinsic)\b', text, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    return "other"


async def one_call(session: httpx.AsyncClient, provider: LLMServerProvider, messages: List[Dict[str, Any]], settings: Dict[str, Any], model: str) -> Tuple[str, str]:
    output, reasoning = await provider.create_chat_completion(
        session=session,
        messages=messages,
        settings=settings,
        model=model,
        thinking_mode=False,
    )
    return output or "", reasoning or ""


async def run_test(args) -> None:
    """Call the local OpenAI-compatible server on port 8080 using model 'hallu'."""
    server_url = f"http://{args.host}:{args.port}"
    provider = LLMServerProvider(server_url)

    context = args.context or input("Context: ")
    prompt = args.prompt or input("Question: ")
    response_text = args.response or input("Response: ")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context: {context}\n\n"
                f"Question: {prompt}\n\n"
                f"Response: {response_text}"
            ),
        },
    ]

    settings = provider.get_provider_default_settings().as_dict()
    settings["max_tokens"] = args.max_tokens
    settings["temperature"] = args.temperature

    counts = Counter()
    errors = 0
    try:
        async with httpx.AsyncClient(timeout=60.0) as session:
            for i in range(max(1, args.repeat)):
                try:
                    if args.vary_seed:
                        settings["seed"] = i + 1
                    output, reasoning = await one_call(session, provider, messages, settings, args.model)
                    label = parse_label(output)
                    counts[label] += 1
                    if not args.quiet:
                        print(f"Run {i+1}: {output.strip()}")
                except Exception as e:
                    errors += 1
                    if not args.quiet:
                        print(f"Run {i+1} failed: {e}")
    except Exception as e:
        print(f"Request failed: {e}")
        return

    total = sum(counts.values())
    print("\n=== Summary ===")
    for key in ("faithful", "intrinsic", "extrinsic", "other"):
        if counts[key] > 0 or key != "other":
            pct = (counts[key] / total * 100) if total else 0.0
            print(f"{key}: {counts[key]} ({pct:.1f}%)")
    if errors:
        print(f"errors: {errors}")


def parse_args():
    p = argparse.ArgumentParser(description="Simple tester for hallucination labeling model")
    p.add_argument("--context", type=str, default=None, help="Context text")
    p.add_argument("--prompt", type=str, default=None, help="Question text")
    p.add_argument("--response", type=str, default=None, help="Model response to evaluate")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    p.add_argument("--model", type=str, default="hallu", help="Model name (default: hallu)")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=10, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    p.add_argument("--repeat", type=int, default=1, help="Number of times to run (e.g., 64)")
    p.add_argument("--quiet", action="store_true", help="Suppress per-run outputs; only print summary")
    p.add_argument("--vary-seed", action="store_true", help="Set different seed per run for diversity")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_test(parse_args()))
