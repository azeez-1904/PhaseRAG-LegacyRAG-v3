#!/usr/bin/env python3
"""
prompt_compressor.py — Prompt Compression Pipeline

PhaseRAG Contribution 2 (MLSys 2027).

Key insight from LegacyRAG v2:
  Prefill dominates wall time for medium/long prompts (84-93%).
  Compressing prompts before inference directly attacks the dominant cost.

Three compression methods:
  1. Extractive: embed sentences with nomic-embed-text, keep top-k by cosine similarity to query
  2. Abstractive: use qwen2:1.5b to summarize retrieved chunks to target length
  3. Token budget: hard truncation to N tokens with sentence-boundary detection

Quality measurement (no external libraries):
  - ROUGE-1 unigram F1 between compressed and original answers
  - Answer length ratio
  - Key entity recall: fraction of named entities from original answer found in compressed answer
"""

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Literal

OLLAMA_URL = "http://127.0.0.1:11434"
RESULTS_DIR = Path(__file__).parent / "results"

CompressionMethod = Literal["extractive", "abstractive", "token_budget"]
CompressionLevel = Literal[0.25, 0.50, 0.75]  # fraction to REMOVE (25% → keep 75%)


# ── Simple math utilities ─────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tokenize_simple(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def rouge1_f1(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-1 F1 between two strings (no external libraries)."""
    hyp_tokens = tokenize_simple(hypothesis)
    ref_tokens = tokenize_simple(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    hyp_set = set(hyp_tokens)
    ref_set = set(ref_tokens)
    overlap = len(hyp_set & ref_set)
    precision = overlap / len(hyp_set) if hyp_set else 0.0
    recall = overlap / len(ref_set) if ref_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_key_entities(text: str, top_n: int = 20) -> set[str]:
    """
    Heuristic key entity extraction: capitalized words/phrases not at sentence start,
    numbers, and technical terms. Used for entity recall measurement.
    """
    entities: set[str] = set()
    words = text.split()
    for i, word in enumerate(words):
        clean = re.sub(r"[^a-zA-Z0-9]", "", word)
        if len(clean) >= 3:
            if clean[0].isupper() and i > 0 and not words[i - 1].endswith("."):
                entities.add(clean.lower())
            if re.match(r"^\d+\.?\d*$", clean):
                entities.add(clean)
            if len(clean) >= 5 and clean.isupper():
                entities.add(clean.lower())
    return set(list(entities)[:top_n])


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _ollama_embed(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    """Get embeddings from Ollama nomic-embed-text."""
    embeddings = []
    for text in texts:
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        embeddings.append(resp["embedding"])
    return embeddings


def _ollama_generate(prompt: str, model: str = "qwen2:1.5b",
                     max_tokens: int = 300) -> tuple[str, float]:
    """Generate text via Ollama."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.1}
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    return resp.get("response", ""), elapsed


# ── Compression methods ───────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at . ! ? boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def compress_extractive(query: str, context: str, keep_fraction: float) -> tuple[str, dict]:
    """
    Keep the top-k sentences most similar to query via nomic-embed-text cosine similarity.
    keep_fraction: 0.25 = keep 25%, 0.50 = keep 50%, 0.75 = keep 75%
    """
    sentences = _split_sentences(context)
    if len(sentences) <= 2:
        return context, {"method": "extractive", "note": "too_short_to_compress",
                         "original_sentences": len(sentences), "kept_sentences": len(sentences)}

    k = max(1, round(len(sentences) * keep_fraction))
    t0 = time.perf_counter()

    # Embed query and all sentences
    all_texts = [query] + sentences
    all_embeddings = _ollama_embed(all_texts)
    query_emb = all_embeddings[0]
    sent_embeddings = all_embeddings[1:]

    # Score each sentence
    scores = [cosine_similarity(query_emb, se) for se in sent_embeddings]
    # Keep top-k sentences in original order
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    top_indices_sorted = sorted(top_indices)
    compressed = " ".join(sentences[i] for i in top_indices_sorted)
    elapsed = time.perf_counter() - t0

    return compressed, {
        "method": "extractive",
        "original_sentences": len(sentences),
        "kept_sentences": k,
        "keep_fraction_target": keep_fraction,
        "compression_latency_ms": round(elapsed * 1000, 2),
        "top_scores": [round(scores[i], 4) for i in top_indices_sorted],
    }


def compress_abstractive(query: str, context: str, keep_fraction: float) -> tuple[str, dict]:
    """
    Use qwen2:1.5b to summarize context to target length.
    """
    original_words = len(context.split())
    target_words = max(20, round(original_words * keep_fraction))
    system_prompt = (
        f"Summarize the following text in approximately {target_words} words, "
        f"preserving the most important information relevant to the question: '{query}'\n\n"
        f"Text to summarize:\n{context}\n\nSummary:"
    )
    t0 = time.perf_counter()
    compressed, gen_elapsed = _ollama_generate(system_prompt, model="qwen2:1.5b",
                                               max_tokens=target_words + 50)
    elapsed = time.perf_counter() - t0

    return compressed.strip(), {
        "method": "abstractive",
        "original_words": original_words,
        "target_words": target_words,
        "actual_words": len(compressed.split()),
        "keep_fraction_target": keep_fraction,
        "compression_latency_ms": round(elapsed * 1000, 2),
        "model": "qwen2:1.5b",
    }


def compress_token_budget(query: str, context: str, keep_fraction: float) -> tuple[str, dict]:
    """
    Hard truncation to N tokens with sentence-boundary detection.
    Estimates tokens as words (rough approximation, avoids tokenizer dependency).
    """
    original_words = len(context.split())
    budget_words = max(10, round(original_words * keep_fraction))
    sentences = _split_sentences(context)

    t0 = time.perf_counter()
    kept: list[str] = []
    word_count = 0
    for sent in sentences:
        sent_words = len(sent.split())
        if word_count + sent_words <= budget_words:
            kept.append(sent)
            word_count += sent_words
        else:
            break
    compressed = " ".join(kept) if kept else sentences[0] if sentences else context
    elapsed = time.perf_counter() - t0

    return compressed, {
        "method": "token_budget",
        "original_words": original_words,
        "budget_words": budget_words,
        "actual_words": len(compressed.split()),
        "keep_fraction_target": keep_fraction,
        "compression_latency_ms": round(elapsed * 1000, 2),
    }


def compress(query: str, context: str, method: CompressionMethod,
             keep_fraction: float) -> tuple[str, dict]:
    """Dispatch to the appropriate compression method."""
    if method == "extractive":
        return compress_extractive(query, context, keep_fraction)
    elif method == "abstractive":
        return compress_abstractive(query, context, keep_fraction)
    elif method == "token_budget":
        return compress_token_budget(query, context, keep_fraction)
    raise ValueError(f"Unknown method: {method}")


def measure_quality(original_answer: str, compressed_answer: str,
                    original_context: str) -> dict:
    """Compute quality metrics comparing answers before and after compression."""
    r1 = rouge1_f1(compressed_answer, original_answer)
    length_ratio = len(compressed_answer.split()) / max(1, len(original_answer.split()))
    orig_entities = extract_key_entities(original_answer)
    comp_entities = extract_key_entities(compressed_answer)
    entity_recall = (
        len(orig_entities & comp_entities) / len(orig_entities)
        if orig_entities else 1.0
    )
    return {
        "rouge1_f1": round(r1, 4),
        "answer_length_ratio": round(length_ratio, 4),
        "entity_recall": round(entity_recall, 4),
        "original_answer_words": len(original_answer.split()),
        "compressed_answer_words": len(compressed_answer.split()),
    }


if __name__ == "__main__":
    query = "What is speculative decoding and why does it improve inference speed?"
    context = (
        "Speculative decoding is a technique for accelerating large language model inference. "
        "It works by using a small draft model to propose candidate token sequences. "
        "The main model then verifies these tokens in a single forward pass. "
        "This allows multiple tokens to be accepted per forward pass, improving throughput. "
        "The speedup depends on the acceptance rate alpha, which measures how often draft "
        "tokens match what the main model would generate. With a high acceptance rate (>70%), "
        "speedups of 2-3x are achievable on hardware with fast parallel verification. "
        "However, on legacy hardware without FP16 tensor cores, the verification step executes "
        "sequentially, negating the parallelism benefit. The NVIDIA Quadro K4200 running the "
        "Vulkan backend is an example of such hardware where speculative decoding provides "
        "no measurable benefit despite reasonable acceptance rates."
    )
    for method in ("extractive", "token_budget", "abstractive"):
        for keep in (0.75, 0.50, 0.25):
            print(f"\n=== {method} keep={keep:.0%} ===")
            compressed, meta = compress(query, context, method, keep)
            print(f"  Meta: {meta}")
            print(f"  Result ({len(compressed.split())} words): {compressed[:200]}...")
