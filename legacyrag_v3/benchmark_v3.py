#!/usr/bin/env python3
"""
benchmark_v3.py — PhaseRAG v3 Experiment Runner

Experiment A: CPU-GPU Heterogeneous Phase Splitting
  - Config 1: CPU-only  (-ngl 0)
  - Config 2: GPU-only  (-ngl 99)
  - Config 3: Phase-split actual (CPU prefill → GPU decode via slot handoff)
  - Config 4: Phase-split theoretical (cpu_prefill_ms + gpu_decode_ms)
  - Config 5: GPU + ngram-simple (v2 best, reference)

Runs 10 prompts (same buckets as v2 for direct comparison).
Saves to results/exp_phase_split.json.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase_splitter import (
    find_model_gguf,
    get_vram,
    run_cpu_only,
    run_gpu_only,
    run_gpu_ngram,
    attempt_slot_handoff,
    compute_theoretical_phase_split,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "exp_phase_split.json"

V2_BEST_TPS = 9.08  # v2 GPU + ngram mean tok/s (reference)
V2_BEST_WALL = 81.4  # v2 GPU + ngram mean wall time

PROMPTS = [
    {"id": 1, "bucket": "short",
     "text": "What is speculative decoding in large language models and why does it improve inference speed?"},
    {"id": 2, "bucket": "short",
     "text": "How does post-training quantization reduce neural network memory footprint while preserving accuracy?"},
    {"id": 3, "bucket": "short",
     "text": "What are the main hardware limitations that make running large language models on consumer GPUs challenging?"},
    {"id": 4, "bucket": "medium",
     "text": (
         "Explain how retrieval-augmented generation (RAG) works and why it is useful for enterprise applications. "
         "Describe the role of embedding models and vector stores in a RAG pipeline. "
         "How is retrieved context injected into the prompt, and what are the main failure modes when the retrieved "
         "chunks are irrelevant or too long? Provide a concrete example with a government records use case where "
         "citizens query a public records database using natural language."
     )},
    {"id": 5, "bucket": "medium",
     "text": (
         "Compare the performance characteristics of modern NVIDIA Ampere or Ada Lovelace GPUs versus legacy Maxwell "
         "architecture GPUs when running transformer model inference. What specific hardware features are absent in "
         "Maxwell that limit throughput? Consider FP16 tensor cores, INT8 dot product support, and memory bandwidth. "
         "How does the NVIDIA Quadro K4200 specifically perform relative to these newer architectures? "
         "What is the practical implication for an organization running open-source LLMs on older hardware?"
     )},
    {"id": 6, "bucket": "medium",
     "text": (
         "What is the difference between greedy decoding, beam search, and temperature-based sampling in language "
         "model text generation? How does each strategy affect output diversity, factual accuracy, and token "
         "generation speed on resource-constrained hardware? Explain the relationship between inference speed "
         "and decoding strategy for a model running at under 2 tokens per second on a legacy Vulkan GPU."
     )},
    {"id": 7, "bucket": "medium",
     "text": (
         "Describe the Open Public Records Act (OPRA) in New Jersey. What obligations does it place on government "
         "agencies regarding document disclosure? How many business days do agencies have to respond to a request? "
         "What categories of records are exempt from disclosure, and what is the role of the Government Records "
         "Council in adjudicating disputes? How might an AI-assisted retrieval system improve compliance efficiency?"
     )},
    {"id": 8, "bucket": "long",
     "text": (
         "You are a research assistant preparing a technical analysis of GPU hardware suitability for large language "
         "model inference. Write a detailed analysis of the NVIDIA Quadro K4200's capabilities and limitations.\n\n"
         "The K4200 is based on the Maxwell architecture (GM204), featuring 1344 CUDA cores, 4GB of GDDR5 VRAM "
         "with 173 GB/s memory bandwidth, and Vulkan 1.3 support via the open-source Mesa/RADV stack. "
         "It lacks hardware FP16 matrix operations, INT8 dot product instructions, and tensor core accelerators "
         "found in Volta and later architectures.\n\n"
         "Address the following in your analysis:\n"
         "1. How do these architectural limitations affect transformer inference throughput?\n"
         "2. What quantization strategies (Q4_K_M, Q2_K, Q8_0) are most practical given 4GB VRAM?\n"
         "3. What is the theoretical maximum tokens per second given the 173 GB/s memory bandwidth constraint "
         "for a 3.8B parameter model at Q4 quantization?\n"
         "4. How does running two such GPUs in a tensor-split configuration via llama.cpp affect throughput?\n"
         "5. What is the practical use case for such hardware in 2025, given that modern alternatives exist?\n\n"
         "Provide specific numbers and calculations where possible."
     )},
    {"id": 9, "bucket": "long",
     "text": (
         "Write a comprehensive technical overview of speculative decoding techniques for accelerating large language "
         "model inference, suitable for inclusion in an IEEE conference paper.\n\n"
         "Cover the following topics in depth:\n"
         "1. The basic mechanism: how a small draft model proposes token sequences and the main model verifies them "
         "in a single forward pass, achieving parallel verification.\n"
         "2. The acceptance rate alpha and how it determines actual speedup versus theoretical maximum speedup.\n"
         "3. N-gram speculative decoding as a draft-model-free alternative.\n"
         "4. Trade-offs between draft model size, acceptance quality, and VRAM overhead on memory-constrained "
         "devices with 4-8GB total VRAM.\n"
         "5. How speculative decoding performs differently when the main model is GPU-resident versus CPU-offloaded.\n"
         "6. Current research challenges: applying speculative decoding to heavily quantized (Q2-Q4) models.\n\n"
         "Include references to relevant literature where appropriate."
     )},
    {"id": 10, "bucket": "long",
     "text": (
         "Analyze the following deployment scenario for an AI-assisted public records retrieval system:\n\n"
         "A mid-size New Jersey municipal government receives approximately 500 OPRA requests per month. "
         "Staff currently spend 2-4 hours per request manually searching physical archives and scanned PDFs. "
         "The IT department has two NVIDIA Quadro K4200 GPUs (4GB VRAM each, Maxwell architecture, Vulkan-only) "
         "available in an existing on-premises server. No new hardware procurement is budgeted. "
         "Staff have no machine learning expertise. Legal compliance requires all data to remain on-premises.\n\n"
         "Design a complete technical architecture for this system addressing:\n"
         "1. Model selection: which open-source LLM and embedding model fit within 8GB combined VRAM?\n"
         "2. Document ingestion pipeline: how to chunk, embed, and index scanned government records.\n"
         "3. Query handling: how the RAG pipeline processes citizen queries and generates OPRA-compliant summaries.\n"
         "4. Expected performance: given 8-9 tok/s on phi3-mini Q4 with dual K4200 Vulkan, what is daily capacity?\n"
         "5. Bottleneck analysis: which components limit throughput and what optimizations are feasible?\n"
         "6. Staff workflow integration: how would municipal employees interact with and validate AI responses?\n"
         "7. Risk assessment: what are the failure modes specific to legacy Vulkan GPU hardware?\n\n"
         "Be specific and pragmatic."
     )},
]


def summarize(results_list: list[dict], key: str = "total_wall_s",
              tps_key: str = "decode_tok_s") -> dict:
    vals = [r[key] for r in results_list if r.get(key) is not None]
    tps_vals = [r[tps_key] for r in results_list if r.get(tps_key) is not None]
    if not vals:
        return {}
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    by_bucket: dict[str, dict] = {}
    for bucket in ("short", "medium", "long"):
        b = [r for r in results_list if r.get("bucket") == bucket and r.get(tps_key) is not None]
        if b:
            b_tps = [r[tps_key] for r in b]
            b_wall = [r[key] for r in b if r.get(key) is not None]
            by_bucket[bucket] = {
                "count": len(b),
                "mean_decode_tok_s": round(sum(b_tps) / len(b_tps), 4),
                "mean_wall_s": round(sum(b_wall) / len(b_wall), 2) if b_wall else None,
            }
    return {
        "n": len(vals),
        "mean_wall_s": round(sum(vals) / n, 2),
        "p95_wall_s": round(sorted_vals[min(int(n * 0.95), n - 1)], 2),
        "mean_decode_tok_s": round(sum(tps_vals) / len(tps_vals), 4) if tps_vals else None,
        "by_bucket": by_bucket,
    }


def run_experiment_a() -> None:
    print("=" * 70)
    print("PhaseRAG v3 — Experiment A: CPU-GPU Heterogeneous Phase Splitting")
    print("=" * 70)

    RESULTS_DIR.mkdir(exist_ok=True)
    model_path = find_model_gguf("phi3/mini")
    print(f"Model: {model_path}\n")

    vram_pre = get_vram()
    print(f"VRAM at start: {vram_pre}\n")

    all_results: list[dict] = []
    slot_handoff_attempted = False
    slot_handoff_success = False
    slot_handoff_failure_reason = None

    for p in PROMPTS:
        print(f"\n{'─'*60}")
        print(f"Prompt {p['id']}/10 | bucket={p['bucket']} | "
              f"~{len(p['text'].split())} words")
        print(f"{'─'*60}")

        record: dict = {
            "id": p["id"],
            "bucket": p["bucket"],
            "prompt_words": len(p["text"].split()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vram_start": get_vram(),
        }

        # ── Config 1: CPU-only ────────────────────────────────────────────────
        print(f"\n  Config 1: CPU-only")
        try:
            cpu_result = run_cpu_only(model_path, p["text"])
            record["cpu_only"] = cpu_result
        except Exception as e:
            print(f"  ERROR cpu_only: {e}", flush=True)
            record["cpu_only"] = {"error": str(e)}

        # ── Config 2: GPU-only ────────────────────────────────────────────────
        print(f"\n  Config 2: GPU-only")
        try:
            gpu_result = run_gpu_only(model_path, p["text"])
            record["gpu_only"] = gpu_result
        except Exception as e:
            print(f"  ERROR gpu_only: {e}", flush=True)
            record["gpu_only"] = {"error": str(e)}

        # ── Config 3: Phase-split theoretical ────────────────────────────────
        if record.get("cpu_only") and record.get("gpu_only") and \
           not record["cpu_only"].get("error") and not record["gpu_only"].get("error"):
            theoretical = compute_theoretical_phase_split(record["cpu_only"], record["gpu_only"])
            record["phase_split_theoretical"] = theoretical
            print(
                f"\n  Config 3 (theoretical): combined={theoretical['theoretical_total_s']:.1f}s  "
                f"speedup_vs_gpu={theoretical.get('speedup_vs_gpu_only', 'N/A')}×",
                flush=True,
            )
        else:
            record["phase_split_theoretical"] = {"error": "missing cpu or gpu result"}

        # ── Config 4: Phase-split actual (slot handoff) — first prompt only ──
        # Attempt on prompt 1; reuse result for all others (handoff is prompt-independent)
        if not slot_handoff_attempted:
            print(f"\n  Config 4: Phase-split slot handoff (Approach A — first attempt)")
            slot_handoff_attempted = True
            try:
                handoff_result = attempt_slot_handoff(model_path, p["text"])
                record["phase_split_actual"] = handoff_result
                if handoff_result.get("mode") == "phase_split_actual":
                    slot_handoff_success = True
                    print(f"  Slot handoff SUCCESS", flush=True)
                else:
                    slot_handoff_failure_reason = handoff_result.get("failure_reason")
                    print(f"  Slot handoff FAILED: {slot_handoff_failure_reason}", flush=True)
            except Exception as e:
                slot_handoff_failure_reason = f"Exception: {type(e).__name__}: {e}"
                record["phase_split_actual"] = {"error": slot_handoff_failure_reason}
                print(f"  Slot handoff ERROR: {slot_handoff_failure_reason}", flush=True)
        else:
            # For subsequent prompts, attempt only if first succeeded
            if slot_handoff_success:
                print(f"\n  Config 4: Phase-split slot handoff (Approach A)")
                try:
                    handoff_result = attempt_slot_handoff(model_path, p["text"])
                    record["phase_split_actual"] = handoff_result
                except Exception as e:
                    record["phase_split_actual"] = {"error": str(e)}
            else:
                record["phase_split_actual"] = {
                    "mode": "phase_split_failed",
                    "failure_reason": f"skipped — initial attempt failed: {slot_handoff_failure_reason}",
                }

        # ── Config 5: GPU + ngram (v2 best reference) ────────────────────────
        print(f"\n  Config 5: GPU + ngram (v2 reference)")
        try:
            ngram_result = run_gpu_ngram(model_path, p["text"])
            record["gpu_ngram"] = ngram_result
        except Exception as e:
            print(f"  ERROR gpu_ngram: {e}", flush=True)
            record["gpu_ngram"] = {"error": str(e)}

        record["vram_end"] = get_vram()
        all_results.append(record)

        # Brief cooldown between prompts
        print(f"\n  [cooldown 10s between prompts]", flush=True)
        time.sleep(10)

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Summarizing results...")

    def extract_config(key: str, results: list[dict]) -> list[dict]:
        out = []
        for r in results:
            cfg = r.get(key, {})
            if cfg and not cfg.get("error"):
                out.append({**cfg, "bucket": r["bucket"], "id": r["id"]})
        return out

    cpu_results = extract_config("cpu_only", all_results)
    gpu_results = extract_config("gpu_only", all_results)
    ngram_results = extract_config("gpu_ngram", all_results)
    theoretical_results = extract_config("phase_split_theoretical", all_results)

    summary = {
        "cpu_only": summarize(cpu_results, "total_wall_s", "decode_tok_s"),
        "gpu_only": summarize(gpu_results, "total_wall_s", "decode_tok_s"),
        "gpu_ngram": summarize(ngram_results, "total_wall_s", "decode_tok_s"),
        "phase_split_theoretical": summarize(
            theoretical_results, "theoretical_total_s", "gpu_decode_tok_s"
        ),
        "slot_handoff": {
            "attempted": slot_handoff_attempted,
            "success": slot_handoff_success,
            "failure_reason": slot_handoff_failure_reason,
        },
        "v2_reference": {
            "mean_tok_s": V2_BEST_TPS,
            "mean_wall_s": V2_BEST_WALL,
            "config": "GPU + ngram-simple, phi3-mini, dual K4200",
        },
    }

    output = {
        "experiment": "exp_phase_split",
        "model": "phi3:mini Q4_K_M",
        "hardware": "dual NVIDIA Quadro K4200, Vulkan 1.3",
        "llama_cpp_build": "b9297",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_prompts": len(PROMPTS),
        "vram_pre_experiment": vram_pre,
        "vram_post_experiment": get_vram(),
        "summary": summary,
        "results": all_results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print("\n=== SUMMARY ===")
    print(f"CPU-only:               {summary['cpu_only'].get('mean_decode_tok_s', 'N/A')} tok/s decode, "
          f"{summary['cpu_only'].get('mean_wall_s', 'N/A')}s mean wall")
    print(f"GPU-only:               {summary['gpu_only'].get('mean_decode_tok_s', 'N/A')} tok/s decode, "
          f"{summary['gpu_only'].get('mean_wall_s', 'N/A')}s mean wall")
    print(f"GPU + ngram:            {summary['gpu_ngram'].get('mean_decode_tok_s', 'N/A')} tok/s decode, "
          f"{summary['gpu_ngram'].get('mean_wall_s', 'N/A')}s mean wall")
    print(f"Phase split theoretical:{summary['phase_split_theoretical'].get('mean_wall_s', 'N/A')}s mean wall")
    print(f"Slot handoff success:   {slot_handoff_success}")
    if not slot_handoff_success:
        print(f"Slot handoff failure:   {slot_handoff_failure_reason}")
    print(f"\nv2 reference (GPU+ngram): {V2_BEST_TPS} tok/s, {V2_BEST_WALL}s mean wall")


if __name__ == "__main__":
    run_experiment_a()
