# PhaseRAG — LegacyRAG v3

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![MLSys 2027](https://img.shields.io/badge/venue-MLSys%202027-blueviolet.svg)](https://mlsys.org/)
[![Series: LegacyRAG](https://img.shields.io/badge/series-LegacyRAG%20v3-orange.svg)](#research-series)

> PhaseRAG investigates CPU-GPU heterogeneous phase splitting for LLM inference: offloading the
> prefill phase (prompt processing) to CPU while reserving GPU for the decode phase (token generation).
> The hypothesis is that CPU BLAS-accelerated prefill outperforms Vulkan-dispatched prefill on Maxwell
> hardware, enabling the GPU's memory bandwidth advantage to be reserved exclusively for decode.
> This repo also contains the complete LegacyRAG v2 benchmark suite as a reference baseline.

---

## What It Does

LLM inference decomposes into two structurally distinct phases:

1. **Prefill** — the model processes all input tokens in a single forward pass, constructing
   the KV cache for the full prompt. This phase is compute-intensive and parallelizable across
   tokens.

2. **Decode** — the model performs autoregressive token generation, emitting one token per
   forward pass. This phase is memory-bandwidth-bound, loading model weights and KV cache on
   every step.

On legacy Maxwell Vulkan hardware (NVIDIA Quadro K4200, no FP16, no tensor cores), a
performance inversion is observed at short prompt lengths: CPU BLAS-accelerated prefill
reaches ~20 tok/s for short inputs, exceeding GPU Vulkan prefill (~13 tok/s) due to Vulkan
dispatch overhead at small batch sizes. GPU decode reaches 8–9 tok/s versus CPU's 5.3 tok/s
because the GPU's 173 GB/s GDDR5 memory bandwidth dominates the weight-loading bottleneck.

**PhaseRAG evaluates whether heterogeneous phase splitting — routing prefill to CPU and decode
to GPU — reduces end-to-end inference latency on this hardware class.**

```
Input prompt + retrieved context (P tokens)
         |
         v
  [CPU llama-server :8082]      prefill via POST /completion, ngl=0
  BLAS-vectorized attention      ~20 tok/s (short); ~1.1 tok/s (medium/long)
         |
         | KV cache state transfer
         | POST /slots/0  <-- HTTP 400: NOT SUPPORTED in llama.cpp b9297
         |       cross-backend KV cache transfer blocked
         |
         v  (theoretical path only)
  [GPU llama-server :8081]      decode via POST /completion, ngl=99
  Vulkan-dispatched attention    ~9.3 tok/s (memory-bandwidth-bound, 173 GB/s GDDR5)
         |
         v
  Streamed token output
```

**Result:** The KV cache transfer fails at runtime (llama.cpp b9297 slot API does not support
cross-backend state transfer). Furthermore, the theoretical speedup for realistic RAG prompts
is negligible — CPU and GPU prefill converge to the same throughput for medium and long
contexts because both become memory-bandwidth-bound at those context lengths. The CPU
advantage is confined to prompts shorter than ~20 tokens.

This repo documents the full experimental investigation, including the v2 baseline suite that
motivated the hypothesis. Target venue: **MLSys 2027**.

---

## Hardware Target

| Component | Spec |
|---|---|
| GPU | 2× NVIDIA Quadro K4200 |
| VRAM | 4 GB GDDR5 per card (8 GB total) |
| Memory bandwidth | 173 GB/s per GPU |
| Architecture | Maxwell GM204 (2014), no FP16, no INT8, no tensor cores |
| Inference backend | llama.cpp b9297 + Vulkan 1.3 |
| CPU | Intel Xeon E5-1620 v3 (4C/8T, 3.5 GHz) |
| RAM | 7.7 GB DDR4 ECC |
| OS | Ubuntu 24.04 LTS |
| LLM | phi3:mini (3.8B, Q4_K_M, ~2.1 GB) |

---

## Key Features

- **`phase_splitter.py`** — CPU→GPU heterogeneous phase splitting implementation. Runs CPU
  prefill via a `-ngl 0` llama-server instance, then attempts KV cache state transfer to a
  GPU decode server via the `/slots/{id}` API. Documents the HTTP 400 failure and theoretical
  fallback measurement.
- **`prompt_compressor.py`** — Three compression methods (extractive sentence similarity,
  abstractive qwen2:1.5b summarization, token-budget hard truncation) with ROUGE-1 F1,
  entity recall, and answer-length-ratio quality measurement.
- **`auto_config.py`** — Hardware-aware model and parameter selection. Detects GPU count and
  total VRAM via `nvidia-smi`, selects the optimal model and `ngl` value, and enables
  ngram speculative decoding automatically when a GPU is present.
- **`benchmark_v3.py`** — Experiment A runner: 10 prompts × 4 configs (CPU-only, GPU-only,
  GPU+ngram, phase-split) with per-request timing breakdowns.
- **`web_ui/`** — FastAPI SSE streaming interface with plain HTML, no JS framework dependency.
  Works in any browser. Displays real-time tok/s and VRAM usage.
- **`install.sh`** — One-script installer: OS detection, hardware detection, Ollama model pull,
  llama-server startup, web UI launch.
- **`legacyrag_v2/`** — Full v2 benchmark code included as a baseline reference. Four
  experiments: GPU baseline, speculative decoding, n-gram, and quantization.

---

## Motivation: v2 Prefill Bottleneck Analysis

LegacyRAG v2 ([IC2E 2026](https://github.com/azeez-1904/LegacyRAG-v2-experiments)) established
that **prefill consumes 80–92% of wall time for medium and long prompts on Maxwell Vulkan**.

| Prompt length | Prefill time | Prefill share of wall time |
|---|---|---|
| Short (18 tok) | 1.65 s | ~7% |
| Medium (78–110 tok) | 124.9 s | **84%** |
| Long (260–367 tok) | 379.3 s | **93%** |

The Vulkan GLSL attention shaders process attention heads without fused kernels, producing
near-quadratic prefill cost at longer contexts. CPU prefill was measured at ~21 tok/s for
short prompts versus GPU's ~11 tok/s — a 2× throughput advantage — motivating the hypothesis
that routing prefill to the CPU could reduce the dominant latency component.

---

## Experiment A: CPU-GPU Phase Splitting

### Architecture Diagram

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                    PhaseRAG Phase Splitter                          │
 │                                                                     │
 │  Incoming prompt (P tokens)                                         │
 │         │                                                           │
 │         ▼                                                           │
 │  ┌─────────────┐    POST /completion     ┌──────────────────────┐  │
 │  │ llama-server │    ngl=0 (CPU-only)     │   CPU prefill        │  │
 │  │  :8082      │◄───────────────────────►│   ~20 tok/s (short)  │  │
 │  │  -ngl 0     │                         │   ~1.1 tok/s (long)  │  │
 │  └─────────────┘                         └──────────────────────┘  │
 │         │                                                           │
 │         │  POST /slots/0                                           │
 │         │  {"action":"save","filename":"slot0.bin"}                 │
 │         │                                                           │
 │         │  ◄── HTTP 400 ── NOT SUPPORTED in b9297 ──────────────── │
 │         │       cross-backend KV cache transfer blocked             │
 │         │                                                           │
 │         │  (theoretical path; cross-backend transfer not supported)  │
 │         ▼                                                           │
 │  ┌─────────────┐    POST /completion     ┌──────────────────────┐  │
 │  │ llama-server │    ngl=99 (GPU)         │   GPU decode         │  │
 │  │  :8081      │◄───────────────────────►│   ~9.3 tok/s         │  │
 │  │  -ngl 99    │    empty prompt +        └──────────────────────┘  │
 │  └─────────────┘    restored KV cache                               │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘
```

### Results Summary

phi3:mini Q4_K_M, dual K4200 Vulkan, llama.cpp b9297, 10 prompts (3 short / 4 medium / 3 long):

| Mode | Mean decode tok/s | Mean wall time | vs GPU-only |
|---|---|---|---|
| GPU-only (ngl=99) | **9.29** | 122.3 s | baseline |
| GPU + ngram-simple | 8.73 | 123.8 s | −6% |
| Phase-split theoretical | 9.29* | 123.0 s | ~0% |
| Phase-split actual | FAILED | — | — |
| CPU-only (ngl=0) | 5.34 | 139.1 s | −43% |

*Decode phase from GPU, prefill from CPU — theoretical composition only.

### Throughput Bar Chart

```
Decode tok/s (mean across 10 prompts, phi3:mini Q4_K_M)

GPU-only      ████████████████████████████████████████████████  9.29
GPU+ngram     █████████████████████████████████████████████     8.73
Phase-split*  ████████████████████████████████████████████████  9.29
CPU-only      ████████████████████████████                      5.34

              0         2         4         6         8        10
              └─────────┴─────────┴─────────┴─────────┴─────────┘

* Theoretical only — cross-backend KV cache transfer failed (HTTP 400)
```

### Prefill Speed by Prompt Length

The CPU prefill advantage is prompt-length dependent — the critical finding:

```
Prefill tok/s by prompt length bucket

SHORT (<20 tok)
  CPU:  ████████████████████  ~20 tok/s
  GPU:  █████████████          ~13 tok/s
  --> CPU wins 1.54×, theoretical phase-split speedup: 1.03×

MEDIUM (70–110 tok)
  CPU:  █                      ~1.1 tok/s
  GPU:  █                      ~1.0 tok/s
  --> No meaningful difference. Theoretical speedup: ~1.0×

LONG (130–340 tok)
  CPU:  █                      ~1.3 tok/s
  GPU:  █                      ~0.9 tok/s
  --> Within noise. Theoretical speedup: ~1.0×

      0     5     10    15    20
      └─────┴─────┴─────┴─────┘

Root cause: both CPU (50 GB/s DDR4) and GPU (173 GB/s GDDR5) become
memory-bandwidth-bound for large KV caches. Neither has a systematic
prefill advantage at realistic RAG prompt lengths.
```

| Prompt length | CPU prefill tok/s | GPU prefill tok/s | Theoretical speedup |
|---|---|---|---|
| Short (n=3, <20 tokens) | **~20 tok/s** | ~13 tok/s | **1.03×** |
| Medium (n=4, 70–110 tokens) | ~1.1 tok/s | ~1.0 tok/s | ~1.00× |
| Long (n=3, 130–340 tokens) | ~1.3 tok/s | ~0.9 tok/s | ~1.00× |

---

## Findings

### Finding 1 — KV cache state transfer fails: HTTP 400

`POST /slots/0` with `{"action": "save", "filename": "..."}` on a CPU llama-server
(b9297 with `--slot-save-path`) returns **HTTP 400**. The llama.cpp slot API is designed for
same-server session continuation — reloading a serialized KV cache on the same backend
instance. It does not support transferring KV cache state between a CPU backend (FP32, host
memory) and a GPU backend (FP32, Vulkan device memory). This is a fundamental architectural
constraint in the current llama.cpp build, not a configuration issue.

### Finding 2 — Theoretical speedup is negligible for realistic RAG prompts

The ~21 tok/s CPU prefill throughput measured in v2 is observed exclusively for prompts
shorter than ~20 tokens, where BLAS SIMD vectorization outperforms Vulkan dispatch overhead.
For medium and long RAG prompts — the representative workload — CPU prefill drops to
1.0–1.4 tok/s, which is statistically indistinguishable from GPU's 0.8–1.3 tok/s. Both
backends are memory-bandwidth-bound at those context lengths. Phase splitting provides a
theoretical speedup of 0.99–1.03× over the actual workload distribution.

### Finding 3 — GPU+ngram underperforms GPU-only in this setup (8.73 vs 9.29 tok/s)

This contradicts v2's +9.7% result. The root cause: ngram speculative decoding builds its
candidate lookup table from tokens generated in the current server session. Each prompt in
this experiment starts a fresh server, so the ngram table is empty for every query. v2's
+9.7% required a sustained multi-prompt session to accumulate prior generated tokens.
Single-query isolation (the correct methodology for measuring cold-start latency) shows no
ngram benefit. v2's improvement reflects batch/sustained use.

### Finding 4 — Memory bandwidth is the true bottleneck

Both CPU (DDR4, ~50 GB/s) and GPU (GDDR5, 173 GB/s) become memory-bandwidth-bound when
the KV cache grows with prompt length. Neither has a systematic prefill advantage at
realistic RAG context lengths. The CPU's BLAS advantage for short prompts disappears once
attention becomes memory-bound rather than compute-bound.

---

## v2 Baseline Results (included in `legacyrag_v2/`)

All experiments: phi3:mini Q4_K_M, dual K4200 Vulkan, b9297.

| Configuration | Mean tok/s | Mean wall (s) | p95 wall (s) | vs v1 |
|---|---|---|---|---|
| v1 (single GPU, Ollama) | 0.95 | 469.4 s | — | baseline |
| exp1: GPU-only baseline | 8.28 | 188.4 s | 410.0 s | **+772%** |
| exp2: Speculative decoding (qwen2 pair) | 3.36 | 112.2 s | 202.1 s | −59% vs exp1 |
| exp3: N-gram (ngram-simple) | **9.08** | **81.4 s** | 191.3 s | **+10%** vs exp1 |
| exp4: qwen2.5-7B Q2_K | 3.82 | 72.3 s | 192.3 s | −54% vs exp1 |

---

## Setup

### Prerequisites

```bash
# llama.cpp b9297 built with Vulkan backend
# llama-server binary at build/bin/llama-server
# Ollama running on port 11434

ollama pull phi3:mini
ollama pull nomic-embed-text
```

### Install

```bash
# Automated (detects hardware, pulls models, starts servers)
bash legacyrag_v3/install.sh

# Manual
pip install -r legacyrag_v3/requirements.txt
```

### Run Phase Split Experiment

```bash
cd legacyrag_v3
python3 benchmark_v3.py
# Results written to results/exp_phase_split.json
```

### Launch Web UI

```bash
uvicorn legacyrag_v3.web_ui.app:app --host 0.0.0.0 --port 8002
# Open http://localhost:8002 in any browser
```

### llama-server (dual-GPU, for v2 baseline replication)

```bash
LD_LIBRARY_PATH=build/bin build/bin/llama-server \
  -m /path/to/phi3-mini.gguf \
  -ngl 99 \
  --split-mode layer \
  --tensor-split 1,1 \
  --main-gpu 0 \
  --port 8081 \
  --slot-save-path /tmp/legacyrag_v3_slots/ \
  --slots

# CPU-only server for phase split prefill
LD_LIBRARY_PATH=build/bin build/bin/llama-server \
  -m /path/to/phi3-mini.gguf \
  -ngl 0 \
  --port 8082 \
  --slot-save-path /tmp/legacyrag_v3_slots/ \
  --slots
```

---

## File Tree

```
PhaseRAG-LegacyRAG-v3/
│
├── legacyrag_v3/                       # v3 PhaseRAG code
│   ├── phase_splitter.py               # CPU→GPU phase splitting (Contribution 1)
│   ├── prompt_compressor.py            # 3 compression methods + quality metrics (Contribution 2)
│   ├── auto_config.py                  # Hardware detection + model/config selection (Contribution 3)
│   ├── benchmark_v3.py                 # Experiment A runner (10 prompts × 4 configs)
│   ├── install.sh                      # One-script hardware-aware installer
│   ├── requirements.txt                # FastAPI, uvicorn, jinja2, python-multipart
│   ├── RESEARCH_LOG.md                 # Dated research log with raw findings
│   ├── results/
│   │   └── exp_phase_split.json        # Experiment A raw results
│   ├── paper_notes/
│   │   └── PhaseRAG_draft.md           # Full paper scaffold (MLSys 2027)
│   └── web_ui/
│       ├── app.py                      # FastAPI SSE streaming server
│       └── templates/
│           └── index.html              # Plain HTML, no JS framework, SSE streaming
│
├── legacyrag_v2/                       # v2 baseline code (IC2E 2026)
│   ├── experiment1_baseline.py         # GPU-only: 8.28 tok/s
│   ├── experiment2_speculative_draft.py # Speculative decoding: 3.36 tok/s (−59%)
│   ├── experiment3_ngram.py            # N-gram: 9.08 tok/s (+10%)
│   ├── experiment4_quantization.py     # qwen2.5-7B Q2_K: 3.82 tok/s (−54%)
│   ├── benchmark_runner.py             # Shared runner infrastructure
│   ├── analysis.py                     # Result aggregation and statistics
│   ├── RESEARCH_LOG.md                 # v2 dated research log
│   ├── results/                        # Raw JSON results per experiment
│   └── paper_notes/
│       └── IC2E_demo_draft.md          # IC2E 2026 demo paper draft
│
├── legacyrag/                          # v1 LegacyRAG pipeline (imported as baseline)
│   ├── vram_scheduler.py               # Per-GPU nvidia-smi monitoring + routing
│   ├── embedder.py                     # nomic-embed-text via Ollama
│   ├── retriever.py                    # Cosine similarity store
│   ├── generator.py                    # llama.cpp streaming client
│   ├── pipeline.py                     # Ingest + query orchestration
│   └── benchmark.py                    # Per-request logging
│
├── main.py                             # FastAPI app (v1 endpoints)
├── requirements.txt                    # v1 dependencies
├── benchmark_results.json              # v1 raw results
├── benchmark_results_baseline.json     # v1 baseline snapshot
├── graphs/                             # Result visualizations
│   ├── latency_breakdown.png
│   ├── tokens_per_second.png
│   ├── vram_usage.png
│   └── scheduler_decisions.png
├── paper_findings.md                   # v1 research summary
├── results_table.csv                   # v1 tabular results
├── schedule_decisions.jsonl            # v1 VRAM scheduler log
├── stress_test_results.json            # v1 stress test output
└── LICENSE
```

---

## Research Series

This repo is the third in the LegacyRAG series, documenting progressive inference optimization
on a fixed hardware platform (Dell Precision Tower 5810, dual NVIDIA Quadro K4200):

| Repo | Venue | Best result | What was tested |
|---|---|---|---|
| [LegacyRAG v1](https://github.com/azeez-1904/LegacyRAG) | arXiv 2026 | 0.95 tok/s | VRAM-aware RAG pipeline; characterized the baseline bottleneck (99.86% latency in generation) |
| [LegacyRAG v2](https://github.com/azeez-1904/LegacyRAG-v2-experiments) | IC2E 2026 | 9.08 tok/s (+772%) | Dual-GPU layer split, speculative decoding, n-gram, quantization |
| **PhaseRAG / LegacyRAG v3** (this repo) | MLSys 2027 | 9.29 tok/s (GPU-only) | CPU-GPU phase splitting, prompt compression, auto-config |
| [TemporalRAG](https://github.com/azeez-1904/TemporalRAG) | ACL 2027 | — | Version-aware document retrieval; temporal consistency in RAG |

### Evolution

```
v1: Single GPU, Ollama
    0.95 tok/s ──────────────────────────────────────────┐
                                                          │
v2: Dual-GPU layer split + n-gram speculative             │
    9.08 tok/s  ──── +772% ───────────────────────────── ┤
                     (b9297, -ngl 99, --spec-type ngram)  │
                                                          │
v3: Phase split investigation                             │
    9.29 tok/s  ──── marginal; KV transfer blocked ─────  ┤
                     (phase split: HTTP 400 in b9297)     │
                                                          │
TemporalRAG: Temporal document versioning                 │
    [see repo] ──────────────────────────────────────── ──┘
```

---

## Novel Contributions (v3)

1. **Phase Splitter** (`legacyrag_v3/phase_splitter.py`) — First systematic study of
   cross-backend CPU→GPU KV cache state transfer in llama.cpp. Documents the HTTP 400 failure
   mode and the conditions under which phase splitting would yield throughput gains (short
   prompts only, <20 tokens). Provides theoretical speedup measurements as a validated upper
   bound.

2. **Prompt Compressor** (`legacyrag_v3/prompt_compressor.py`) — Three compression
   strategies (extractive, abstractive, token-budget) with automated quality measurement
   (ROUGE-1 F1, entity recall, answer-length ratio). Designed for VRAM-constrained
   deployments where no separate compression model can be loaded.

3. **Auto-Config** (`legacyrag_v3/auto_config.py`) — Hardware detection that auto-selects
   optimal model, `ngl`, and speculative decoding settings for detected GPU configuration.
   Enables zero-config deployment for institutions without ML expertise.

---

## Citation

```bibtex
@misc{phaserag2027,
  title   = {PhaseRAG: CPU-GPU Heterogeneous Phase Splitting for LLM Inference on Legacy Hardware},
  author  = {Ahmad, Azeez},
  year    = {2027},
  url     = {https://github.com/azeez-1904/PhaseRAG-LegacyRAG-v3}
}
```

For the v2 baseline results, please also cite:

```bibtex
@misc{legacyrag_v2_2026,
  title   = {LegacyRAG v2: Speculative Decoding and Quantization on Maxwell Vulkan},
  author  = {Ahmad, Azeez},
  year    = {2026},
  url     = {https://github.com/azeez-1904/LegacyRAG-v2-experiments}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
