#!/usr/bin/env python3
"""
experiment_b_rerun.py — Rerun Experiment B for P5-P10 (Vulkan TDR fix)

Root cause of crash: vk::DeviceLostError when Ollama (qwen2:1.5b + nomic-embed)
and llama-server ran on the same GPUs simultaneously. Maxwell Vulkan has no fault
isolation between concurrent GPU processes — documented as a hardware finding.

Fix: Two strict phases with no GPU overlap.
  Phase 1 — Precompute: CPU-only Ollama (OLLAMA_NUM_GPU=0) runs all
             extractive + abstractive compressions. llama-server NOT running.
             Results saved to results/compressed_p5_p10.json.
  Phase 2 — Inference: Ollama killed. llama-server starts. Runs all
             generation calls using precomputed compressed contexts.
             Token-budget computed inline (no Ollama needed).

After both phases, merges with existing P1-P4 results from exp_compression.json
into results/exp_compression_full.json.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompt_compressor import compress, measure_quality, rouge1_f1
from phase_splitter import find_model_gguf, get_vram

PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "build_b9297"
RESULTS_DIR = Path(__file__).parent / "results"
COMPRESSED_FILE = RESULTS_DIR / "compressed_p5_p10.json"
FULL_RESULTS_FILE = RESULTS_DIR / "exp_compression_full.json"
EXISTING_FILE = RESULTS_DIR / "exp_compression.json"

SERVER_PORT = 8083
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MAX_TOKENS = 150
METHODS = ["token_budget", "extractive", "abstractive"]
KEEP_FRACTIONS = [0.75, 0.50, 0.25]
OLLAMA_URL = "http://127.0.0.1:11434"

# Prompts P5-P10 (medium + long from experiment_b_compression.py)
PROMPTS_P5_P10 = [
    {
        "id": 5, "bucket": "medium",
        "query": "What are the main factors limiting LLM inference speed on the NVIDIA Quadro K4200?",
        "context": (
            "The NVIDIA Quadro K4200 is based on the Maxwell GM204 architecture (2014), featuring 1344 CUDA "
            "cores, 4 GB GDDR5 VRAM, and 173 GB/s memory bandwidth. It supports Vulkan 1.3 through the "
            "proprietary NVIDIA driver stack. Unlike later architectures, Maxwell lacks several features "
            "critical for efficient LLM inference:\n\n"
            "First, Maxwell has no FP16 matrix multiply units. All matrix operations execute as FP32 SIMD "
            "instructions, halving the theoretical throughput compared to hardware with native FP16 support. "
            "Second, there are no INT8 dot product instructions, eliminating 8-bit quantization acceleration. "
            "Third, Maxwell predates tensor cores entirely, which in Volta+ architectures provide 4-8x speedup "
            "for small matrix multiplications common in attention computation.\n\n"
            "Memory bandwidth (173 GB/s) is the primary throughput determinant for autoregressive decode. "
            "A 3.8B parameter model at Q4_K_M (2.1 GB) requires reading the full weight matrix each forward "
            "pass, theoretically limiting throughput to approximately 173/2.1 approximately 82 passes per second. However, "
            "KV cache and Vulkan dispatch overhead reduce practical throughput to 8-9 tok/s, representing "
            "roughly 10% efficiency versus the bandwidth-bound theoretical maximum. For prefill with longer "
            "contexts, the attention computation adds O(n squared) memory accesses, making prefill throughput "
            "approximately 0.7 tok/s for 300-400 token prompts on this hardware."
        ),
    },
    {
        "id": 6, "bucket": "medium",
        "query": "What records are exempt from disclosure under New Jersey OPRA?",
        "context": (
            "The Open Public Records Act establishes a presumption of openness for government records but "
            "enumerates specific categories exempt from mandatory disclosure. The major exemption categories "
            "include:\n\n"
            "Personnel records: Employee performance evaluations, disciplinary records, medical information, "
            "and home addresses of government employees are generally exempt to protect privacy.\n\n"
            "Criminal investigative records: Records compiled for law enforcement purposes that would interfere "
            "with an ongoing investigation, deprive defendants of fair trial rights, disclose confidential "
            "sources, or endanger the safety of individuals are exempt under N.J.S.A. 47:1A-3.\n\n"
            "Attorney-client privileged communications: Legal advice provided by government counsel and "
            "attorney work product prepared in anticipation of litigation are exempt.\n\n"
            "Trade secrets and proprietary business information: Information submitted by private entities "
            "to government agencies that constitutes trade secrets or confidential commercial information.\n\n"
            "Security-sensitive information: Records that, if disclosed, would jeopardize the security of "
            "any structure, facility, or system, including cybersecurity vulnerabilities and access codes.\n\n"
            "Draft documents: Preliminary drafts that are not circulated outside the agency and advisory, "
            "consultative, or deliberative materials that are part of the decision-making process."
        ),
    },
    {
        "id": 7, "bucket": "medium",
        "query": "How does llama.cpp distribute a model across multiple GPUs using Vulkan?",
        "context": (
            "llama.cpp implements multi-GPU tensor splitting through its Vulkan backend when multiple Vulkan "
            "devices are present. The -ngl (number of GPU layers) flag controls how many transformer layers "
            "are offloaded to GPU. When set to 99 (or any value exceeding total layers), all available layers "
            "are distributed across detected Vulkan devices.\n\n"
            "The distribution algorithm divides transformer layers approximately evenly by VRAM capacity across "
            "detected GPUs. For two GPUs with equal VRAM (e.g., dual K4200 at 4 GB each), layers split "
            "approximately 50/50. For unequal VRAM configurations, the split is weighted proportionally. "
            "Layer boundaries are chosen at transformer block boundaries to minimize inter-GPU communication.\n\n"
            "In practice, dual K4200 operation places phi3-mini's 32 transformer layers across both cards: "
            "GPU0 handles layers 0-15 (using approximately 1976 MB) and GPU1 handles layers 16-31 "
            "(approximately 1545 MB). The asymmetry arises because the embedding and output projection layers "
            "reside on GPU0. During a forward pass, activations transfer between GPUs at each layer boundary "
            "via PCIe 3.0 x16. The PCIe transfer overhead is small relative to compute time on Maxwell. "
            "This dual-GPU split reduces per-GPU VRAM pressure, eliminating thermal throttling observed in "
            "single-GPU operation and yielding 8.28 tok/s versus 0.95 tok/s in the prior single-GPU baseline."
        ),
    },
    {
        "id": 8, "bucket": "long",
        "query": "Design a RAG pipeline for processing OPRA requests in a New Jersey municipality.",
        "context": (
            "A retrieval-augmented generation (RAG) pipeline for municipal OPRA compliance involves several "
            "interconnected components. The following architecture is derived from analysis of government "
            "document retrieval requirements and the constraints of on-premises deployment.\n\n"
            "Document Ingestion Layer: Municipal records arrive in multiple formats - scanned PDFs from "
            "physical archives, digitized meeting minutes, budget spreadsheets, and email correspondence. "
            "A preprocessing pipeline applies OCR (Tesseract or equivalent) to scanned documents, extracts "
            "text from digital PDFs using PyMuPDF, and normalizes formatting. Documents are chunked into "
            "512-token segments with 50-token overlap to preserve sentence context at boundaries. Each chunk "
            "is tagged with metadata: document type, date range, department, and OPRA exemption flags.\n\n"
            "Embedding and Indexing: Each text chunk is embedded using nomic-embed-text (274 MB, 768-dimension "
            "vectors) running via the Ollama API on the local GPU. Embeddings are stored in a NumPy array "
            "alongside chunk text and metadata. For the K4200 hardware, embedding throughput is approximately "
            "50-100 chunks per minute. A corpus of 10,000 scanned pages produces roughly 20,000-30,000 chunks "
            "requiring 2-3 hours of initial indexing. The vector store occupies approximately 450 MB on disk "
            "(20,000 chunks x 768 dimensions x 4 bytes FP32).\n\n"
            "Query Processing: When a citizen submits an OPRA request, the query text is embedded using the "
            "same nomic-embed-text model. The top-k most similar chunks (k=5 by default) are retrieved via "
            "cosine similarity against the stored embeddings. Retrieval latency is under 100ms for 30,000 "
            "vectors on CPU. Retrieved chunks are concatenated into a context window of up to 2048 tokens.\n\n"
            "Generation: The assembled prompt (system instruction + retrieved context + query) is passed to "
            "phi3-mini running on dual K4200 GPUs via llama-server. For a 400-token prompt, generation of "
            "a 200-token response takes approximately 25 seconds for decode at 8 tok/s, plus 280 seconds "
            "for prefill at 0.7 tok/s - dominated by prefill. The generated response summarizes relevant "
            "records and flags any OPRA exemptions applicable to the request.\n\n"
            "Staff Workflow: Generated responses are presented to the records custodian as a draft, not a "
            "final determination. The custodian reviews the AI summary, verifies the cited records, and "
            "either approves the draft response or adds annotations before sending to the requestor. All "
            "AI-generated responses are logged with the source chunks and similarity scores for audit "
            "purposes. The system does not autonomously release records - it accelerates the search and "
            "drafting process while maintaining human oversight required by OPRA.\n\n"
            "Performance Constraints: At 0.95 tok/s (v1 baseline) or 8-9 tok/s (v2 optimized), response "
            "generation for a typical 200-word OPRA response requires 25-470 seconds excluding prefill. "
            "With 500 requests per month and assuming 8-hour working days, the system must process roughly "
            "25 requests per day. At 9 tok/s decode with 400-token prompts, each request requires "
            "approximately 5-8 minutes. Daily capacity is approximately 60-96 requests, comfortably "
            "exceeding the 25-request daily demand."
        ),
    },
    {
        "id": 9, "bucket": "long",
        "query": "Compare Maxwell, Pascal, and Ampere GPU architectures for LLM inference workloads.",
        "context": (
            "NVIDIA GPU architectures from Maxwell (2014) through Ampere (2020) span a critical transition "
            "period for AI workloads. The following comparison focuses on features relevant to transformer "
            "model inference.\n\n"
            "Maxwell (GM200/GM204, 2014-2016): Introduced asynchronous compute and improved memory "
            "compression. Maxwell lacks FP16 matrix multiply units, INT8 dot products, and tensor cores. "
            "All matrix operations execute as FP32 SIMD instructions on CUDA cores. Memory bandwidth on "
            "the K4200 (GM204) is 173 GB/s with 4 GB GDDR5. Under Vulkan (for systems without CUDA "
            "support), throughput for 3.8B parameter models is approximately 8-9 tok/s decode and "
            "0.7 tok/s prefill for long contexts. Maxwell is the current boundary of practical LLM "
            "inference on CUDA-abandoned hardware using Vulkan backends.\n\n"
            "Pascal (GP100/GP102, 2016-2018): Introduced FP16 storage and computation (though not "
            "native FP16 matrix multiply on consumer cards). Pascal added NVLink for multi-GPU "
            "communication and introduced unified memory improvements. The GTX 1080 Ti provides "
            "484 GB/s memory bandwidth and 11 GB GDDR5X, yielding approximately 35-45 tok/s for "
            "3.8B parameter models via CUDA. Pascal is widely used for budget LLM inference and "
            "remains fully supported by CUDA 12.x.\n\n"
            "Volta (V100, 2017): First architecture with Tensor Cores - 4x4 matrix multiply-accumulate "
            "units operating at FP16 with FP32 accumulation. Tensor cores enable approximately 8x "
            "throughput improvement for matrix multiplications versus CUDA cores. The V100 achieves "
            "900 GB/s HBM2 bandwidth. For LLM inference, V100 delivers 150-200 tok/s for 7B models.\n\n"
            "Turing (RTX 2000 series, 2018): Added INT8 and INT4 tensor cores for quantized inference. "
            "Consumer Turing cards (RTX 2080 Ti) provide 616 GB/s GDDR6 bandwidth. INT8 inference on "
            "Turing yields approximately 2x throughput over FP16 for the same model.\n\n"
            "Ampere (A100/RTX 3000 series, 2020): Third-generation tensor cores with BF16 support, "
            "sparsity acceleration (2x speedup for 50% sparse models), and NVLink 3.0. The A100 "
            "provides 2 TB/s HBM2e bandwidth. For LLM inference, A100 achieves 600-800 tok/s for "
            "7B models at FP16, representing approximately 70-100x improvement over Maxwell Vulkan "
            "for the same workload. Ada Lovelace (RTX 4000 series, 2022) adds FP8 support and "
            "transformer engine acceleration, further widening the gap."
        ),
    },
    {
        "id": 10, "bucket": "long",
        "query": "What are the deployment risks of using legacy Vulkan GPUs for production LLM inference?",
        "context": (
            "Deploying LLM inference on legacy Vulkan GPU hardware (Maxwell, Kepler architectures) "
            "introduces several categories of risk that organizations must evaluate before committing "
            "to this approach for production workloads.\n\n"
            "Hardware Reliability Risks: Consumer and workstation GPUs from 2013-2016 are operating "
            "near or beyond their design lifecycle. Mean time between failures for GPU hardware "
            "increases significantly after 7-10 years of operation. Thermal paste degradation, "
            "capacitor aging, and fan bearing wear are common failure modes. Unlike modern AI "
            "accelerators in enterprise settings, replacement units may be unavailable or expensive "
            "on the secondary market. Production deployments should maintain spare GPU inventory "
            "and implement failover to CPU-only inference.\n\n"
            "Driver and Software Support Risks: NVIDIA's proprietary driver support for Maxwell "
            "(compute capability 5.0/5.2) is maintained as of 2025 but may be deprecated in future "
            "driver releases. Vulkan support for Maxwell depends on driver versions 390+ which remain "
            "available on Ubuntu 20.04/22.04/24.04. llama.cpp's Vulkan backend is under active "
            "development and API compatibility may break between builds. The b5576 to b9297 transition "
            "removed the --draft-max flag and changed tokenizer validation behavior - build-specific "
            "testing is essential before upgrading.\n\n"
            "Performance Degradation Risks: Thermal throttling on Maxwell hardware causes significant "
            "tok/s variance across requests (observed range 0.29-1.66 tok/s in LegacyRAG v1 single-GPU "
            "operation). Dual-GPU operation mitigates this by halving per-GPU thermal load, but ambient "
            "temperature and airflow remain variables. Production deployments should monitor GPU "
            "temperature and implement request queuing to prevent thermal runaway under sustained load.\n\n"
            "Vulkan TDR (Timeout Detection and Recovery): Under sustained concurrent GPU workloads, "
            "Maxwell Vulkan drivers may raise vk::DeviceLostError - the GPU kernel exceeds the driver "
            "timeout and the device is reset. This was observed during LegacyRAG v3 Experiment B when "
            "qwen2:1.5b (Ollama) and phi3-mini (llama-server) ran concurrently on the same dual K4200 "
            "system. Maxwell has no process isolation for Vulkan compute queues - all processes share "
            "the same command scheduler. A DeviceLost error terminates all Vulkan processes and requires "
            "driver reset. Production deployments must serialize GPU access strictly.\n\n"
            "Security and Compliance Risks: On-premises deployment provides data sovereignty benefits "
            "but introduces physical security requirements. AI-generated responses must be reviewed by "
            "qualified staff before legal use.\n\n"
            "Capacity Risks: At 8-9 tok/s decode on dual K4200, a 200-token response requires "
            "approximately 25 seconds. With 400-token prompts, total wall time including prefill reaches "
            "5-8 minutes per request at single-request throughput."
        ),
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def start_cpu_ollama() -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OLLAMA_NUM_GPU"] = "0"
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  CPU-only Ollama PID {proc.pid}, waiting...", flush=True)
    for _ in range(20):
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
                if r.status == 200:
                    print("  Ollama ready (CPU-only mode).", flush=True)
                    return proc
        except Exception:
            pass
    raise TimeoutError("Ollama did not start within 40s")


def stop_ollama(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(5)
    print("  Ollama stopped.", flush=True)


def token_budget_compress(query: str, context: str, keep: float) -> tuple[str, dict]:
    from prompt_compressor import compress_token_budget
    return compress_token_budget(query, context, keep)


def start_llama_server(model_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(BIN_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        str(BIN_DIR / "llama-server"),
        "-m", str(model_path),
        "-ngl", "99",
        "--port", str(SERVER_PORT),
        "--host", "127.0.0.1",
        "--ctx-size", "4096",
        "--threads", "4",
        "--parallel", "1",
        "--cache-prompt",
        "--log-disable",
    ]
    log = open(RESULTS_DIR / "server_expb_rerun.log", "a")
    log.write(f"\n=== server start {datetime.now(timezone.utc).isoformat()} ===\n")
    log.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
    print(f"  llama-server PID {proc.pid}, waiting...", flush=True)
    for _ in range(60):
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
                if r.status == 200:
                    print("  llama-server healthy.", flush=True)
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited early")
    raise TimeoutError("llama-server not healthy within 120s")


def stop_llama_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("  llama-server stopped.", flush=True)
    time.sleep(10)


def run_completion(prompt: str) -> tuple[dict, float]:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": 0.1,
        "stop": ["</s>", "<|end|>"],
        "stream": False,
        "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as r:
        raw = r.read()
    return json.loads(raw), time.perf_counter() - t0


def build_prompt(query: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def extract_timings(resp: dict, wall_s: float) -> dict:
    t = resp.get("timings", {})
    pn = t.get("prompt_n", 0)
    pm = t.get("prompt_ms", 0)
    dn = t.get("predicted_n", 0)
    dm = t.get("predicted_ms", 0)
    return {
        "prefill_n": pn,
        "prefill_ms": round(pm, 2),
        "prefill_tok_s": round(pn / (pm / 1000), 3) if pm > 0 else None,
        "decode_n": dn,
        "decode_ms": round(dm, 2),
        "decode_tok_s": round(t.get("predicted_per_second", dn / (dm / 1000) if dm > 0 else 0), 3),
        "wall_s": round(wall_s, 3),
    }


# ── Phase 1: Precompute compressions ─────────────────────────────────────────

def phase1_precompute() -> dict:
    print("\n" + "=" * 70)
    print("PHASE 1: Precomputing all compressions (CPU-only Ollama, no llama-server)")
    print("=" * 70)

    ollama_proc = start_cpu_ollama()
    all_compressed: dict = {}

    try:
        for p in PROMPTS_P5_P10:
            print(f"\n  Prompt {p['id']} ({p['bucket']}, {len(p['context'].split())} words)")
            all_compressed[p["id"]] = {}

            for method in METHODS:
                for keep in KEEP_FRACTIONS:
                    label = f"{method}_keep{int(keep*100)}pct"
                    try:
                        t0 = time.perf_counter()
                        compressed, meta = compress(p["query"], p["context"], method, keep)
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        meta["compression_latency_ms"] = round(elapsed_ms, 2)
                        actual_ratio = len(compressed.split()) / max(1, len(p["context"].split()))
                        all_compressed[p["id"]][label] = {
                            "method": method,
                            "keep_fraction_target": keep,
                            "actual_ratio": round(actual_ratio, 4),
                            "compressed_context": compressed,
                            "compression_meta": meta,
                            "compressed_words": len(compressed.split()),
                        }
                        print(f"    {label}: {len(p['context'].split())}→{len(compressed.split())} words "
                              f"({actual_ratio:.2f}) in {elapsed_ms:.0f}ms", flush=True)
                    except Exception as e:
                        all_compressed[p["id"]][label] = {"error": str(e)}
                        print(f"    {label}: ERROR {e}", flush=True)
    finally:
        stop_ollama(ollama_proc)

    with open(COMPRESSED_FILE, "w") as f:
        json.dump(all_compressed, f, indent=2)
    print(f"\n  Compressed contexts saved to {COMPRESSED_FILE}")
    return all_compressed


# ── Phase 2: Inference (no Ollama running) ────────────────────────────────────

def phase2_inference(all_compressed: dict, model_path: Path) -> list[dict]:
    print("\n" + "=" * 70)
    print("PHASE 2: Inference (llama-server only, Ollama NOT running)")
    print("=" * 70)

    server_proc = start_llama_server(model_path)
    time.sleep(3)
    results = []

    try:
        for p in PROMPTS_P5_P10:
            print(f"\n{'─'*60}")
            print(f"Prompt {p['id']}/10 | {p['bucket']} | {len(p['context'].split())} words")
            print(f"{'─'*60}")

            record: dict = {
                "id": p["id"],
                "bucket": p["bucket"],
                "query": p["query"],
                "context_words": len(p["context"].split()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vram_start": get_vram(),
            }

            # Baseline
            print(f"\n  [baseline] uncompressed...", flush=True)
            try:
                resp, wall_s = run_completion(build_prompt(p["query"], p["context"]))
                baseline_answer = resp.get("content", "")
                bt = extract_timings(resp, wall_s)
                record["baseline"] = {**bt,
                                       "context_words": len(p["context"].split()),
                                       "answer_words": len(baseline_answer.split()),
                                       "answer_preview": baseline_answer[:150]}
                print(f"    prefill={bt['prefill_tok_s']} tok/s  "
                      f"decode={bt['decode_tok_s']} tok/s  wall={wall_s:.1f}s", flush=True)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                record["baseline"] = {"error": str(e)}
                baseline_answer = ""
                results.append(record)
                continue

            # Variants
            variants = []
            p_compressed = all_compressed.get(p["id"], {})

            for method in METHODS:
                for keep in KEEP_FRACTIONS:
                    label = f"{method}_keep{int(keep*100)}pct"
                    print(f"\n  [{label}]", flush=True)

                    comp_data = p_compressed.get(label, {})
                    if comp_data.get("error"):
                        variants.append({"label": label, "method": method,
                                         "keep_fraction": keep,
                                         "error": f"compression failed: {comp_data['error']}"})
                        continue

                    compressed_context = comp_data.get("compressed_context", "")
                    if not compressed_context:
                        variants.append({"label": label, "method": method,
                                         "keep_fraction": keep, "error": "no compressed context"})
                        continue

                    comp_words = comp_data.get("compressed_words", 0)
                    actual_ratio = comp_data.get("actual_ratio", 0)
                    print(f"    context: {len(p['context'].split())}→{comp_words} words "
                          f"({actual_ratio:.2f} ratio)", flush=True)

                    try:
                        resp, wall_s = run_completion(
                            build_prompt(p["query"], compressed_context)
                        )
                        compressed_answer = resp.get("content", "")
                        timings = extract_timings(resp, wall_s)
                        quality = measure_quality(baseline_answer, compressed_answer, p["context"])
                        print(f"    prefill={timings['prefill_tok_s']} tok/s  "
                              f"decode={timings['decode_tok_s']} tok/s  wall={wall_s:.1f}s  "
                              f"ROUGE={quality['rouge1_f1']:.3f}  ent={quality['entity_recall']:.3f}",
                              flush=True)
                        variants.append({
                            "label": label,
                            "method": method,
                            "keep_fraction_target": keep,
                            "actual_ratio": actual_ratio,
                            "compression_meta": comp_data.get("compression_meta", {}),
                            **timings,
                            "quality": quality,
                            "answer_preview": compressed_answer[:150],
                        })
                    except Exception as e:
                        print(f"    inference ERROR: {e}", flush=True)
                        variants.append({"label": label, "method": method,
                                         "keep_fraction": keep,
                                         "compression_meta": comp_data.get("compression_meta", {}),
                                         "error": str(e)})

            record["variants"] = variants
            record["vram_end"] = get_vram()
            results.append(record)
            # Brief pause between prompts to prevent TDR
            print(f"  [5s pause before next prompt]", flush=True)
            time.sleep(5)

    finally:
        stop_llama_server(server_proc)

    return results


# ── Merge and summarize ───────────────────────────────────────────────────────

def merge_and_summarize(new_results: list[dict]) -> None:
    print(f"\n{'='*70}\nMerging with P1-P4 results...")

    existing = {}
    if EXISTING_FILE.exists():
        with open(EXISTING_FILE) as f:
            existing = json.load(f)

    all_results = list(existing.get("results", [])) + new_results
    all_results.sort(key=lambda r: r["id"])

    def mean(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    rows = []
    for r in all_results:
        b = r.get("baseline", {})
        for v in r.get("variants", []):
            if v.get("error"):
                continue
            q = v.get("quality", {})
            cm = v.get("compression_meta", {})
            rows.append({
                "id": r["id"],
                "bucket": r["bucket"],
                "method": v["method"],
                "keep_fraction": v.get("keep_fraction_target"),
                "actual_ratio": v.get("actual_ratio"),
                "comp_latency_ms": cm.get("compression_latency_ms"),
                "rouge1_f1": q.get("rouge1_f1"),
                "entity_recall": q.get("entity_recall"),
                "answer_length_ratio": q.get("answer_length_ratio"),
                "wall_s": v.get("wall_s"),
                "decode_tok_s": v.get("decode_tok_s"),
                "baseline_wall_s": b.get("wall_s"),
            })

    agg: dict = {}
    for row in rows:
        key = (row["method"], row["keep_fraction"])
        if key not in agg:
            agg[key] = {k: [] for k in
                        ["rouge1_f1", "entity_recall", "wall_s", "decode_tok_s",
                         "comp_latency_ms", "actual_ratio", "answer_length_ratio"]}
        for k in agg[key]:
            agg[key][k].append(row.get(k))

    # By bucket
    bucket_agg: dict = {}
    for row in rows:
        key = (row["method"], row["keep_fraction"], row["bucket"])
        if key not in bucket_agg:
            bucket_agg[key] = {"rouge1_f1": [], "wall_s": [], "actual_ratio": []}
        for k in bucket_agg[key]:
            bucket_agg[key][k].append(row.get(k))

    summary_rows = []
    for (method, keep), vals in sorted(agg.items()):
        summary_rows.append({
            "method": method,
            "keep_fraction": keep,
            "n_samples": len([v for v in vals["wall_s"] if v is not None]),
            "mean_actual_ratio": mean(vals["actual_ratio"]),
            "mean_comp_latency_ms": mean(vals["comp_latency_ms"]),
            "mean_rouge1_f1": mean(vals["rouge1_f1"]),
            "mean_entity_recall": mean(vals["entity_recall"]),
            "mean_answer_length_ratio": mean(vals["answer_length_ratio"]),
            "mean_wall_s": mean(vals["wall_s"]),
            "mean_decode_tok_s": mean(vals["decode_tok_s"]),
        })

    by_bucket_rows = []
    for (method, keep, bucket), vals in sorted(bucket_agg.items()):
        by_bucket_rows.append({
            "method": method, "keep_fraction": keep, "bucket": bucket,
            "n": len([v for v in vals["wall_s"] if v is not None]),
            "mean_rouge1_f1": mean(vals["rouge1_f1"]),
            "mean_wall_s": mean(vals["wall_s"]),
            "mean_actual_ratio": mean(vals["actual_ratio"]),
        })

    output = {
        "experiment": "exp_compression_full",
        "model": "phi3:mini Q4_K_M",
        "hardware": "dual NVIDIA Quadro K4200, Vulkan 1.3",
        "llama_cpp_build": "b9297",
        "max_tokens": MAX_TOKENS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_prompts_total": len(all_results),
        "vdr_tdr_note": (
            "vk::DeviceLostError (Vulkan TDR) occurred during initial run when "
            "qwen2:1.5b (Ollama) and phi3-mini (llama-server) ran concurrently on "
            "dual K4200. Maxwell Vulkan has no process isolation for compute queues. "
            "Fix: precompute all compressions with CPU-only Ollama before starting "
            "llama-server. Documented as hardware reliability finding."
        ),
        "summary": summary_rows,
        "by_bucket": by_bucket_rows,
        "results": all_results,
    }

    with open(FULL_RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved to {FULL_RESULTS_FILE}")

    print("\n=== FULL SUMMARY TABLE ===")
    print(f"{'Method':<14} {'Keep':>5} {'N':>3} {'Ratio':>6} {'CompMs':>7} "
          f"{'ROUGE1':>7} {'EntRec':>7} {'Wall':>7}s {'Decode':>7}")
    print("─" * 75)
    for row in summary_rows:
        print(
            f"{row['method']:<14} {row['keep_fraction']:>5.0%} {row['n_samples']:>3} "
            f"{row['mean_actual_ratio']:>6.3f} {(row['mean_comp_latency_ms'] or 0):>7.0f} "
            f"{(row['mean_rouge1_f1'] or 0):>7.3f} {(row['mean_entity_recall'] or 0):>7.3f} "
            f"{(row['mean_wall_s'] or 0):>8.1f} {(row['mean_decode_tok_s'] or 0):>7.2f}"
        )

    print("\n=== BY BUCKET ===")
    for bucket in ("short", "medium", "long"):
        print(f"\n  {bucket.upper()}:")
        brows = [r for r in by_bucket_rows if r["bucket"] == bucket and r["n"] > 0]
        for r in sorted(brows, key=lambda x: (x["method"], x["keep_fraction"])):
            print(f"    {r['method']:<14} {r['keep_fraction']:>5.0%}: "
                  f"ROUGE={r['mean_rouge1_f1']:.3f}  wall={r['mean_wall_s']:.1f}s  "
                  f"ratio={r['mean_actual_ratio']:.3f}  n={r['n']}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    model_path = find_model_gguf("phi3/mini")
    print(f"Model: {model_path}")
    print(f"VRAM: {get_vram()}")

    # Phase 1: precompute (CPU Ollama only)
    if COMPRESSED_FILE.exists():
        print(f"\nFound existing {COMPRESSED_FILE}, loading...")
        with open(COMPRESSED_FILE) as f:
            all_compressed = json.load(f)
        # Convert string keys back to int
        all_compressed = {int(k): v for k, v in all_compressed.items()}
    else:
        all_compressed = phase1_precompute()
        all_compressed = {int(k): v for k, v in all_compressed.items()}

    # Phase 2: inference (llama-server only)
    new_results = phase2_inference(all_compressed, model_path)

    # Merge and summarize
    merge_and_summarize(new_results)
    print("\nDone.")
