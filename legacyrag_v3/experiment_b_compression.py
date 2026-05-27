#!/usr/bin/env python3
"""
experiment_b_compression.py — Prompt Compression Experiment (PhaseRAG v3)

10 RAG-style prompts (query + retrieved context), 3 methods × 3 compression levels.
Server starts ONCE for all inferences to allow b9297 KV cache reuse.
Baseline (uncompressed) answer stored per prompt; quality measured against it.

Methods:  extractive (nomic-embed-text cosine), abstractive (qwen2:1.5b), token_budget
Levels:   keep 75%, 50%, 25% of context (i.e. remove 25%, 50%, 75%)
Metrics:  compression_ratio, compression_latency_ms, rouge1_f1, entity_recall,
          answer_length_ratio, total_wall_s (inference only), tok_per_sec
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompt_compressor import compress, measure_quality, rouge1_f1
from phase_splitter import find_model_gguf, get_vram, _start_server, _stop_server

PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "build_b9297"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "exp_compression.json"
SERVER_PORT = 8082
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MAX_TOKENS = 150
METHODS = ["token_budget", "extractive", "abstractive"]
KEEP_FRACTIONS = [0.75, 0.50, 0.25]

# Override port in phase_splitter imports
import phase_splitter as _ps
_ps.SERVER_PORT = SERVER_PORT
_ps.SERVER_URL = SERVER_URL

# ── RAG-style prompts: query + retrieved context ───────────────────────────────
# Context = simulated retrieved document chunks. Query = user question.
# Combined prompt = context + "\n\nQuestion: " + query + "\n\nAnswer:"

PROMPTS = [
    # ── Short context (~80-100 words) ──────────────────────────────────────────
    {
        "id": 1, "bucket": "short",
        "query": "How many days does a New Jersey government agency have to respond to an OPRA request?",
        "context": (
            "The Open Public Records Act (OPRA), N.J.S.A. 47:1A-1 et seq., requires New Jersey government "
            "agencies to respond to public records requests within seven business days of receipt. If additional "
            "time is required, the custodian must provide written notice within seven business days stating the "
            "reason for the delay and the anticipated date of fulfillment. Agencies that fail to respond within "
            "the statutory period are deemed to have denied the request. Citizens may appeal denials to the "
            "Government Records Council or file a complaint in Superior Court."
        ),
    },
    {
        "id": 2, "bucket": "short",
        "query": "What does Q4_K_M quantization mean for an LLM model file?",
        "context": (
            "GGUF quantization formats encode neural network weights at reduced precision to decrease memory "
            "footprint. Q4_K_M denotes 4-bit quantization using the K-quant method with medium-size blocks. "
            "In practice, a 3.8 billion parameter model quantized to Q4_K_M occupies approximately 2.1 GB on "
            "disk, compared to 7.6 GB at full FP32 precision. The K-quant scheme groups weights into blocks and "
            "applies mixed-precision encoding, typically using 4 bits for most weights and 6 bits for salient "
            "outlier weights. This balances memory savings with minimal accuracy degradation versus 8-bit formats."
        ),
    },
    {
        "id": 3, "bucket": "short",
        "query": "What is the role of the Government Records Council in New Jersey?",
        "context": (
            "The New Jersey Government Records Council (GRC) is a state agency established under OPRA to "
            "adjudicate disputes between requestors and government agencies over public records access. The GRC "
            "provides a free, administrative alternative to Superior Court litigation. Citizens who believe their "
            "OPRA request was improperly denied may file a complaint with the GRC within 45 days of the denial. "
            "The GRC investigates complaints, issues findings, and can order agencies to provide records and pay "
            "attorney fees to prevailing requestors. GRC decisions may be appealed to the Appellate Division."
        ),
    },
    # ── Medium context (~250-300 words) ───────────────────────────────────────
    {
        "id": 4, "bucket": "medium",
        "query": "How does speculative decoding improve token generation speed?",
        "context": (
            "Speculative decoding is an inference acceleration technique for autoregressive language models "
            "introduced by Leviathan et al. (ICML 2023) and Chen et al. (2023). The core mechanism involves "
            "two models: a small, fast draft model and the larger target model. The draft model autoregressively "
            "generates a sequence of k candidate tokens. The target model then verifies all k+1 positions "
            "(including the original token) in a single parallel forward pass. Tokens that match the target "
            "model's distribution are accepted; the first rejected token causes truncation at that position.\n\n"
            "The expected speedup is governed by the draft acceptance rate alpha (α), defined as the probability "
            "that a draft token matches the target distribution. With α=0.7 and k=8 draft tokens, the expected "
            "number of accepted tokens per verification step is approximately 3.7, compared to 1 for standard "
            "autoregressive decoding — a theoretical 3.7× speedup. In practice, speedup depends on whether the "
            "target model can process k+1 tokens in parallel faster than generating them sequentially.\n\n"
            "On hardware with FP16 tensor cores (e.g., NVIDIA A100), the verification step runs approximately "
            "k× faster than k sequential forward passes due to batched matrix multiplication. On Maxwell Vulkan "
            "hardware without FP16 support, verification executes as k+1 sequential FP32 operations, eliminating "
            "the parallelism benefit. Empirical results from LegacyRAG v2 show 36.9% mean acceptance rate but "
            "no throughput improvement on dual K4200 Vulkan, confirming the hardware dependency of speculative "
            "decoding gains. N-gram speculative decoding avoids the draft model entirely by predicting "
            "continuations from previously generated context, yielding +9.7% on the same hardware."
        ),
    },
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
            "Third, Maxwell predates tensor cores entirely, which in Volta+ architectures provide 4-8× speedup "
            "for small matrix multiplications common in attention computation.\n\n"
            "Memory bandwidth (173 GB/s) is the primary throughput determinant for autoregressive decode. "
            "A 3.8B parameter model at Q4_K_M (2.1 GB) requires reading the full weight matrix each forward "
            "pass, theoretically limiting throughput to approximately 173/2.1 ≈ 82 passes per second. However, "
            "KV cache and Vulkan dispatch overhead reduce practical throughput to 8-9 tok/s, representing "
            "roughly 10% efficiency versus the bandwidth-bound theoretical maximum. For prefill with longer "
            "contexts, the attention computation adds O(n²) memory accesses, making prefill throughput "
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
            "devices are present. The `-ngl` (number of GPU layers) flag controls how many transformer layers "
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
    # ── Long context (~500-600 words) ─────────────────────────────────────────
    {
        "id": 8, "bucket": "long",
        "query": "Design a RAG pipeline for processing OPRA requests in a New Jersey municipality.",
        "context": (
            "A retrieval-augmented generation (RAG) pipeline for municipal OPRA compliance involves several "
            "interconnected components. The following architecture is derived from analysis of government "
            "document retrieval requirements and the constraints of on-premises deployment.\n\n"
            "Document Ingestion Layer: Municipal records arrive in multiple formats — scanned PDFs from "
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
            "(20,000 chunks × 768 dimensions × 4 bytes FP32).\n\n"
            "Query Processing: When a citizen submits an OPRA request, the query text is embedded using the "
            "same nomic-embed-text model. The top-k most similar chunks (k=5 by default) are retrieved via "
            "cosine similarity against the stored embeddings. Retrieval latency is under 100ms for 30,000 "
            "vectors on CPU. Retrieved chunks are concatenated into a context window of up to 2048 tokens.\n\n"
            "Generation: The assembled prompt (system instruction + retrieved context + query) is passed to "
            "phi3-mini running on dual K4200 GPUs via llama-server. For a 400-token prompt, generation of "
            "a 200-token response takes approximately 25 seconds for decode at 8 tok/s, plus 280 seconds "
            "for prefill at 0.7 tok/s — dominated by prefill. The generated response summarizes relevant "
            "records and flags any OPRA exemptions applicable to the request.\n\n"
            "Staff Workflow: Generated responses are presented to the records custodian as a draft, not a "
            "final determination. The custodian reviews the AI summary, verifies the cited records, and "
            "either approves the draft response or adds annotations before sending to the requestor. All "
            "AI-generated responses are logged with the source chunks and similarity scores for audit "
            "purposes. The system does not autonomously release records — it accelerates the search and "
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
            "Volta (V100, 2017): First architecture with Tensor Cores — 4×4 matrix multiply-accumulate "
            "units operating at FP16 with FP32 accumulation. Tensor cores enable approximately 8× "
            "throughput improvement for matrix multiplications versus CUDA cores. The V100 achieves "
            "900 GB/s HBM2 bandwidth. For LLM inference, V100 delivers 150-200 tok/s for 7B models.\n\n"
            "Turing (RTX 2000 series, 2018): Added INT8 and INT4 tensor cores for quantized inference. "
            "Consumer Turing cards (RTX 2080 Ti) provide 616 GB/s GDDR6 bandwidth. INT8 inference on "
            "Turing yields approximately 2× throughput over FP16 for the same model.\n\n"
            "Ampere (A100/RTX 3000 series, 2020): Third-generation tensor cores with BF16 support, "
            "sparsity acceleration (2× speedup for 50% sparse models), and NVLink 3.0. The A100 "
            "provides 2 TB/s HBM2e bandwidth. For LLM inference, A100 achieves 600-800 tok/s for "
            "7B models at FP16, representing approximately 70-100× improvement over Maxwell Vulkan "
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
            "removed the --draft-max flag and changed tokenizer validation behavior — build-specific "
            "testing is essential before upgrading.\n\n"
            "Performance Degradation Risks: Thermal throttling on Maxwell hardware causes significant "
            "tok/s variance across requests (observed range 0.29-1.66 tok/s in LegacyRAG v1 single-GPU "
            "operation). Dual-GPU operation mitigates this by halving per-GPU thermal load, but ambient "
            "temperature and airflow remain variables. Production deployments should monitor GPU "
            "temperature and implement request queuing to prevent thermal runaway under sustained load.\n\n"
            "Security and Compliance Risks: On-premises deployment on legacy hardware provides data "
            "sovereignty benefits (required for OPRA, HIPAA, and similar compliance frameworks) but "
            "introduces physical security requirements. The server must be physically secured, patched "
            "against OS vulnerabilities, and isolated from public networks. AI-generated responses must "
            "be reviewed by qualified staff before legal use — the system's output is a draft aid, "
            "not a legally binding determination.\n\n"
            "Capacity Risks: At 8-9 tok/s decode on dual K4200, a 200-token response requires "
            "approximately 25 seconds. With 400-token prompts (including RAG context), total wall time "
            "including prefill reaches 5-8 minutes per request. Concurrent requests are not supported "
            "in the llama-server parallel=1 configuration. Organizations expecting more than 100 "
            "requests per day should evaluate whether the system can meet SLA requirements given "
            "queue depth and response time expectations."
        ),
    },
]


def run_completion(prompt: str) -> tuple[dict, float]:
    import urllib.request
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
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read()
    elapsed = time.perf_counter() - t0
    return json.loads(raw), elapsed


def build_prompt(query: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


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


def run_experiment_b():
    print("=" * 70)
    print("PhaseRAG v3 — Experiment B: Prompt Compression Pipeline")
    print("=" * 70)
    RESULTS_DIR.mkdir(exist_ok=True)

    model_path = find_model_gguf("phi3/mini")
    print(f"Model: {model_path}")
    vram_pre = get_vram()
    print(f"VRAM at start: {vram_pre}\n")

    # Start server ONCE for all inferences
    print("Starting llama-server (ngl=99, single session for all inferences)...")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(BIN_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")
    import urllib.request as _ur
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
    log_path = RESULTS_DIR / "server_expb.log"
    log_file = open(log_path, "a")
    log_file.write(f"\n=== Exp B server start {datetime.now(timezone.utc).isoformat()} ===\n")
    log_file.flush()
    server_proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    print(f"  Server PID {server_proc.pid}, waiting for health...", flush=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(2)
        try:
            with _ur.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
                if r.status == 200:
                    print("  Server healthy.\n", flush=True)
                    break
        except Exception:
            pass
        if server_proc.poll() is not None:
            raise RuntimeError(f"Server exited early. Check {log_path}")
    time.sleep(2)

    all_results = []

    try:
        for p in PROMPTS:
            print(f"\n{'─'*60}")
            print(f"Prompt {p['id']}/10 | bucket={p['bucket']}")
            print(f"  Query: {p['query'][:70]}...")
            print(f"  Context: {len(p['context'].split())} words")
            print(f"{'─'*60}")

            record: dict = {
                "id": p["id"],
                "bucket": p["bucket"],
                "query": p["query"],
                "context_words": len(p["context"].split()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # ── Baseline: uncompressed ────────────────────────────────────────
            print(f"\n  [baseline] uncompressed ({len(p['context'].split())} words)...")
            baseline_prompt = build_prompt(p["query"], p["context"])
            try:
                resp, wall_s = run_completion(baseline_prompt)
                baseline_answer = resp.get("content", "")
                baseline_timings = extract_timings(resp, wall_s)
                record["baseline"] = {
                    **baseline_timings,
                    "context_words": len(p["context"].split()),
                    "answer_words": len(baseline_answer.split()),
                    "answer_preview": baseline_answer[:150],
                }
                print(
                    f"    prefill={baseline_timings['prefill_tok_s']} tok/s  "
                    f"decode={baseline_timings['decode_tok_s']} tok/s  wall={wall_s:.1f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"    ERROR baseline: {e}", flush=True)
                record["baseline"] = {"error": str(e)}
                baseline_answer = ""
                all_results.append(record)
                continue

            # ── Compressed variants ───────────────────────────────────────────
            variants = []
            for method in METHODS:
                for keep in KEEP_FRACTIONS:
                    label = f"{method}_keep{int(keep*100)}pct"
                    print(f"\n  [{label}]", flush=True)

                    # Compress
                    try:
                        t_comp_start = time.perf_counter()
                        compressed_context, comp_meta = compress(
                            p["query"], p["context"], method, keep
                        )
                        comp_elapsed = time.perf_counter() - t_comp_start
                        comp_meta["compression_latency_ms"] = round(comp_elapsed * 1000, 2)
                        comp_words = len(compressed_context.split())
                        orig_words = len(p["context"].split())
                        actual_ratio = comp_words / orig_words if orig_words > 0 else 1.0
                        print(
                            f"    compressed: {orig_words}→{comp_words} words "
                            f"({actual_ratio:.2f} ratio) in {comp_elapsed*1000:.0f}ms",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"    compression ERROR: {e}", flush=True)
                        variants.append({"label": label, "method": method,
                                         "keep_fraction": keep, "error": str(e)})
                        continue

                    # Infer
                    compressed_prompt = build_prompt(p["query"], compressed_context)
                    try:
                        resp, wall_s = run_completion(compressed_prompt)
                        compressed_answer = resp.get("content", "")
                        timings = extract_timings(resp, wall_s)
                        quality = measure_quality(baseline_answer, compressed_answer, p["context"])
                        print(
                            f"    prefill={timings['prefill_tok_s']} tok/s  "
                            f"decode={timings['decode_tok_s']} tok/s  wall={wall_s:.1f}s  "
                            f"ROUGE-1={quality['rouge1_f1']:.3f}  entity_recall={quality['entity_recall']:.3f}",
                            flush=True,
                        )
                        variants.append({
                            "label": label,
                            "method": method,
                            "keep_fraction_target": keep,
                            "actual_ratio": round(actual_ratio, 4),
                            "compression_meta": comp_meta,
                            **timings,
                            "quality": quality,
                            "answer_preview": compressed_answer[:150],
                        })
                    except Exception as e:
                        print(f"    inference ERROR: {e}", flush=True)
                        variants.append({"label": label, "method": method,
                                         "keep_fraction": keep,
                                         "compression_meta": comp_meta, "error": str(e)})

            record["variants"] = variants
            all_results.append(record)

    finally:
        print("\nStopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\nSummarizing...")

    rows = []
    for r in all_results:
        b = r.get("baseline", {})
        for v in r.get("variants", []):
            if v.get("error"):
                continue
            q = v.get("quality", {})
            rows.append({
                "bucket": r["bucket"],
                "method": v["method"],
                "keep_fraction": v.get("keep_fraction_target"),
                "actual_ratio": v.get("actual_ratio"),
                "comp_latency_ms": v.get("compression_meta", {}).get("compression_latency_ms"),
                "rouge1_f1": q.get("rouge1_f1"),
                "entity_recall": q.get("entity_recall"),
                "answer_length_ratio": q.get("answer_length_ratio"),
                "wall_s": v.get("wall_s"),
                "decode_tok_s": v.get("decode_tok_s"),
                "baseline_wall_s": b.get("wall_s"),
            })

    def mean(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    agg: dict = {}
    for row in rows:
        key = (row["method"], row["keep_fraction"])
        if key not in agg:
            agg[key] = {"rouge1_f1": [], "entity_recall": [], "wall_s": [],
                        "decode_tok_s": [], "comp_latency_ms": [], "actual_ratio": []}
        for k in agg[key]:
            agg[key][k].append(row.get(k))

    summary_rows = []
    for (method, keep), vals in sorted(agg.items()):
        summary_rows.append({
            "method": method,
            "keep_fraction": keep,
            "mean_actual_ratio": mean(vals["actual_ratio"]),
            "mean_comp_latency_ms": mean(vals["comp_latency_ms"]),
            "mean_rouge1_f1": mean(vals["rouge1_f1"]),
            "mean_entity_recall": mean(vals["entity_recall"]),
            "mean_wall_s": mean(vals["wall_s"]),
            "mean_decode_tok_s": mean(vals["decode_tok_s"]),
        })

    output = {
        "experiment": "exp_compression",
        "model": "phi3:mini Q4_K_M",
        "hardware": "dual NVIDIA Quadro K4200, Vulkan 1.3",
        "llama_cpp_build": "b9297",
        "max_tokens": MAX_TOKENS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vram_pre": vram_pre,
        "vram_post": get_vram(),
        "summary": summary_rows,
        "results": all_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Method':<15} {'Keep':>6} {'Ratio':>7} {'CompMs':>8} {'ROUGE1':>7} {'EntRec':>7} {'Wall':>7}s {'Decode':>8} tok/s")
    print("─" * 75)
    for row in summary_rows:
        print(
            f"{row['method']:<15} {row['keep_fraction']:>6.0%} "
            f"{row['mean_actual_ratio']:>7.3f} {row['mean_comp_latency_ms']:>8.0f} "
            f"{row['mean_rouge1_f1']:>7.3f} {row['mean_entity_recall']:>7.3f} "
            f"{row['mean_wall_s']:>8.1f} {row['mean_decode_tok_s']:>8.2f}"
        )


if __name__ == "__main__":
    run_experiment_b()
