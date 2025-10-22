#!/usr/bin/env python3
"""
Test DeepSeek API with the OpenAI-compatible Python SDK.

Prereqs
-------
1) pip install openai
2) export DEEPSEEK_API_KEY="sk-..."
   (Optional) export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # default is this

Examples
--------
# Simple non-streaming call
python test_deepseek_openai.py -p "用三句话解释量子叠加"

# Streaming tokens to the console
python test_deepseek_openai.py -p "写一首关于清晨的五言绝句" --stream

# Choose model explicitly
python test_deepseek_openai.py -p "Explain transformers like I'm five" -m deepseek-chat
python test_deepseek_openai.py -p "Refactor this code" -m deepseek-coder
python test_deepseek_openai.py -p "Chain-of-thought *disabled*, only final answer." -m deepseek-reasoner

Notes
-----
- This script uses the OpenAI-compatible Chat Completions API.
- If you see authentication errors, double-check DEEPSEEK_API_KEY and any firewall/proxy settings.
- For programmatic use, import the functions `chat_once` and `chat_stream` into your project.
"""

import argparse
import os
import sys
import time
from typing import Optional

try:
    # OpenAI Python SDK >= 1.0
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    print("[ERROR] Failed to import openai. Try: pip install openai", file=sys.stderr)
    raise


DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def create_client(
    api_key: Optional[str] = "sk-5d1d1180313c45589472a340afe4a5f5",
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> OpenAI:
    """Create an OpenAI-compatible client for DeepSeek."""
    key = api_key
    if not key:
        print("[ERROR] Missing DEEPSEEK_API_KEY environment variable.", file=sys.stderr)
        sys.exit(2)

    base = base_url or DEFAULT_BASE_URL

    # The OpenAI client accepts base_url and api_key for 3rd-party compatible backends
    client = OpenAI(api_key=key, base_url=base, timeout=timeout)
    return client


def chat_once(
    client: OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
) -> str:
    """Send a single chat completion request and return the assistant text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    dt = (time.perf_counter() - t0) * 1000

    text = resp.choices[0].message.content or ""

    # Optional: print usage metrics if provided by backend
    usage_info = getattr(resp, "usage", None)
    if usage_info:
        prompt_tokens = getattr(usage_info, "prompt_tokens", None)
        completion_tokens = getattr(usage_info, "completion_tokens", None)
        total_tokens = getattr(usage_info, "total_tokens", None)
        print(
            f"\n[usage] prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}",
            file=sys.stderr,
        )

    print(f"[latency] {dt:.1f} ms", file=sys.stderr)
    return text


def chat_stream(
    client: OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
) -> None:
    """Stream tokens to stdout."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True,
    )

    # The OpenAI SDK yields chunk objects; print content deltas as they arrive
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                print(delta.content, end="", flush=True)
        except Exception:
            # Some backends may emit non-standard chunks; ignore gracefully
            pass

    dt = (time.perf_counter() - t0) * 1000
    print()  # final newline
    print(f"[latency] {dt:.1f} ms", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test DeepSeek API (OpenAI compatible)")
    p.add_argument("-p", "--prompt", type=str, required=True, help="User prompt text")
    p.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help="Model name (e.g., deepseek-chat, deepseek-coder, deepseek-reasoner)")
    p.add_argument("--system", type=str, default=None, help="Optional system message")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL for DeepSeek API")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    p.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=512, help="Max new tokens")
    p.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling p")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    client = create_client(base_url=args.base_url, timeout=args.timeout)

    if args.stream:
        chat_stream(
            client=client,
            prompt=args.prompt,
            model=args.model,
            system=args.system,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
    else:
        text = chat_once(
            client=client,
            prompt=args.prompt,
            model=args.model,
            system=args.system,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        print(text)


if __name__ == "__main__":
    main()
