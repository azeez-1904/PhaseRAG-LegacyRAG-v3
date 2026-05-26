# PhaseRAG: CPU-GPU Heterogeneous Phase Splitting for LLM Inference on CUDA-Abandoned Hardware

**Target Venue:** MLSys 2027
**Submission Deadline:** TBD (~early 2027)
**Format:** Full Research Paper (10 pages, MLSys double-column)
**GitHub:** https://github.com/azeez-1904/LegacyRAG-v3
**Authors:** Azeez Shaik, NJIT

---

## Abstract

_[PLACEHOLDER — fill after Experiment A and B complete]_

We present PhaseRAG, a CPU-GPU heterogeneous inference system that routes large language
model (LLM) phases to whichever compute substrate executes them faster. On legacy Maxwell
Vulkan hardware (NVIDIA Quadro K4200, no FP16/tensor cores), CPU prefill runs at ~21 tok/s
versus GPU prefill at ~0.7 tok/s — a 30× gap — while GPU decode at ~8-9 tok/s outpaces
CPU decode at ~6.5 tok/s by ~30%. Our LegacyRAG v2 system achieved 8.28 tok/s mean
throughput with prefill consuming 84–93% of total wall time for medium/long prompts,
suggesting that routing prefill to CPU could reduce end-to-end latency by up to [v3_result]×.
We further contribute a prompt compression pipeline that attacks the prefill bottleneck
directly, and a zero-configuration deployment framework that auto-selects models and
parameters for any GPU hardware class.

**Keywords:** heterogeneous inference, CPU-GPU scheduling, speculative decoding,
quantization, legacy GPU, Vulkan, RAG, edge computing

---

## I. Introduction

The majority of research on LLM inference optimization targets data-center hardware: NVIDIA
A100/H100 with tensor cores, NVLink, and FP16/BF16 matrix multiply. Yet many organizations
operate fleets of pre-2017 GPU workstations — Maxwell, Pascal, or Kepler architecture —
that cannot run CUDA-based LLM frameworks due to driver and compute-capability constraints.
For these users, llama.cpp's Vulkan backend is the only viable path to on-device LLM
inference without hardware replacement.

Our prior work (LegacyRAG v1 [1], v2 [2]) characterized the performance profile of the
NVIDIA Quadro K4200 (Maxwell GM204, 4GB GDDR5, 173 GB/s, Vulkan 1.3) running phi3-mini
at Q4_K_M quantization:

| v2 Key Result | Value |
|---|---|
| GPU decode throughput (dual K4200 + ngram) | 9.08 tok/s |
| GPU prefill throughput (medium/long prompts) | 0.62–0.95 tok/s |
| CPU prefill throughput (measured) | ~21 tok/s |
| CPU decode throughput (measured) | ~6.5 tok/s |
| Prefill % of total wall time (long prompts) | 84–93% |

**The PhaseRAG Insight:** The bottleneck is phase-specific. CPU is 30× faster at prefill;
GPU is 30% faster at decode. A system that dynamically routes each phase to the faster device
could theoretically eliminate the prefill bottleneck while retaining GPU decode throughput.

**Research Questions:**

1. Can KV cache state be transferred between CPU and GPU llama.cpp backends with low latency?
2. What is the theoretical and measured speedup from CPU prefill + GPU decode phase splitting?
3. How much additional speedup does prompt compression provide by reducing prefill cost?
4. What deployment abstractions are needed for zero-config deployment on legacy hardware?

**Contributions:**

1. **PhaseRAG Phase Splitter:** CPU-GPU heterogeneous inference routing with KV cache handoff
   analysis, including the first systematic study of cross-backend slot save/restore in llama.cpp.
2. **Prompt Compression Pipeline:** Three compression strategies (extractive, abstractive,
   token-budget) with quality measurement, benchmarked on Maxwell Vulkan hardware.
3. **Zero-Config Deployment:** Hardware-aware model selection and one-script installer
   for organizations without ML expertise.

---

## II. Related Work

### A. Heterogeneous LLM Inference

**Splitwise (Kachris et al., ISCA 2024) [3]** proposes prefill-decode disaggregation across
separate compute nodes in data-center settings. PhaseRAG applies the same insight to a
single-node CPU+GPU system on legacy hardware, where the phase performance inversion is
more extreme than in data-center deployments.

**BatOpt (TCC 2024) [4]** optimizes batching strategies for heterogeneous GPU clusters.
PhaseRAG targets heterogeneous CPU+GPU within a single node, a different scheduling problem.

**Trihinas et al. (IC2E 2024) [5]** analyze edge inference deployment constraints.
PhaseRAG provides empirical data for the Maxwell Vulkan class specifically.

### B. Speculative Decoding

**Leviathan et al. (ICML 2023) [6]** introduced speculative decoding with draft models.
Our v2 results showed this provides no speedup on Maxwell Vulkan due to sequential FP32
verification — a negative result that constrains the applicability of this technique.
N-gram speculative decoding (llama.cpp `--spec-type ngram-simple`) provides +9.7%
consistently and is the recommended alternative for legacy hardware.

### C. Quantization

**Dettmers et al. (NeurIPS 2022) [7]** demonstrated 8-bit matrix quantization. LLM.int4()
and GGUF Q2_K represent further compression. Our v2 results showed 7B Q2_K (3.82 tok/s)
is 54% slower than 3.8B Q4 (8.28 tok/s) on K4200, confirming parameter count as the binding
constraint on 173 GB/s bandwidth.

### D. Prompt Compression

**LLMLingua (Jiang et al., EMNLP 2023)** uses a smaller LLM to compress prompts. PhaseRAG
implements a simpler extractive approach using embedding similarity — no compression model
needed — appropriate for VRAM-constrained deployments.

---

## III. System Architecture

### A. Hardware Baseline

| Component | Specification |
|---|---|
| GPUs (×2) | NVIDIA Quadro K4200, Maxwell GM204, 2014 |
| VRAM | 4 GB GDDR5 × 2 = 8 GB total |
| Bandwidth | 173 GB/s per GPU |
| Vulkan | 1.3, no FP16/INT8/tensor cores |
| CPU | Intel Xeon E5-1620 v3, 4C/8T, 3.50 GHz |
| RAM | 7.7 GB DDR4 ECC |
| OS | Ubuntu 24.04 LTS |

### B. Software Stack

| Component | Version / Source |
|---|---|
| llama.cpp | b9297, Vulkan backend |
| Inference server | llama-server, dual-GPU split (`-ngl 99`) |
| Embedding model | nomic-embed-text via Ollama |
| Main LLM | phi3-mini 3.8B Q4_K_M |
| Compression LLM | qwen2:1.5b Q4_K_M (abstractive only) |
| Python | 3.12, stdlib only for experiment scripts |

### C. v2 Baseline Performance (Reference)

_From LegacyRAG v2 [2], all configurations use phi3-mini Q4_K_M on dual K4200 Vulkan._

| Configuration | Mean tok/s | Mean wall (s) | p95 wall (s) |
|---|---|---|---|
| v1 (single GPU, Ollama) | 0.95 | 469.4 | — |
| exp1: GPU-only baseline | 8.28 | 188.4 | 410.0 |
| exp2: Speculative decoding | 3.36 | 112.2 | 202.1 |
| exp3: N-gram speculative | **9.08** | **81.4** | 191.3 |
| exp4: 7B Q2_K quantization | 3.82 | 72.3 | 192.3 |

---

## IV. PhaseRAG: CPU-GPU Phase Splitting

### A. Phase Performance Asymmetry on Maxwell Vulkan

The fundamental observation enabling PhaseRAG: prefill and decode have opposite device
preferences on Maxwell Vulkan hardware.

**CPU prefill advantage:**
Maxwell Vulkan lacks fused attention kernels. The Vulkan backend processes each attention
head sequentially in FP32. For large context windows, this means O(n²) attention operations
at FP32 throughput — extremely slow. The CPU, by contrast, benefits from SIMD vectorization,
better cache behavior for sequential reads, and llama.cpp's optimized BLAS routines.
Measured CPU prefill: ~21 tok/s vs GPU prefill: 0.7 tok/s for long prompts.

**GPU decode advantage:**
Autoregressive token generation is memory-bandwidth-bound. Each forward pass reads all
model weights (2.1 GB for phi3-mini Q4) plus the KV cache. The K4200 at 173 GB/s
outperforms the CPU's memory bandwidth (~50 GB/s for DDR4 quad-channel) by ~3.4×.
Measured: GPU decode 8-9 tok/s vs CPU decode 6.5 tok/s.

**Theoretical combined latency:**
For a prompt of P tokens generating G tokens:

```
CPU-only:        T_cpu = P/21 + G/6.5  (seconds)
GPU-only:        T_gpu = P/0.7 + G/8.5 (seconds)
Phase-split:     T_ps  = P/21 + t_handoff + G/8.5 (seconds)

Speedup vs GPU-only:
  S = T_gpu / T_ps = (P/0.7 + G/8.5) / (P/21 + t_handoff + G/8.5)
```

For a long prompt (P=400, G=200): T_gpu = 595s, T_ps = 42.9s (assuming t_handoff ≈ 1s),
theoretical speedup ≈ **13.9×**.

_[PLACEHOLDER — fill with measured Experiment A results]_

### B. KV Cache Handoff Implementation

_[PLACEHOLDER — fill after Experiment A determines which approach works]_

**Approach A — Slot Save/Restore:**
llama.cpp b9297 exposes `--slot-save-path` and a `/slots/{id}` POST API for saving and
restoring server slot state (which includes the KV cache). The attempted workflow:
1. CPU server (`-ngl 0`) prefills prompt, saves slot to `/tmp/legacyrag_v3_slots/slot0.bin`
2. GPU server (`-ngl 99`) restores slot, decodes with empty prompt continuation

**Technical challenge:** CPU KV cache is stored in FP32 in host memory; GPU KV cache is
stored in FP32 in Vulkan device memory. The llama.cpp slot serialization format may or may
not be compatible between backends.

**Approach B — Theoretical Measurement (fallback):**
If slot handoff fails, run CPU-only and GPU-only separately, extract `timings.prompt_ms`
and `timings.predicted_ms`, and compute `theoretical_combined = cpu_prefill_ms + gpu_decode_ms`.
This is a valid contribution: it precisely quantifies the potential speedup even if the
engineering implementation is blocked by backend incompatibility.

### C. Experiment A Results

_[PLACEHOLDER — fill from results/exp_phase_split.json after experiment completes]_

| Configuration | Prefill tok/s | Decode tok/s | Mean wall (s) | vs v2 best |
|---|---|---|---|---|
| CPU-only | [result] | [result] | [result] | [result] |
| GPU-only | [result] | [result] | [result] | [result] |
| GPU + ngram (v2 reference) | ~0.7 | ~9.1 | 81.4 | 1.0× |
| Phase split (theoretical) | [CPU] | [GPU] | [result] | [result]× |
| Phase split (actual, if feasible) | [CPU] | [GPU] | [result] | [result]× |

---

## V. Prompt Compression Results

_[PLACEHOLDER — fill from results/exp_compression.json after Experiment B completes]_

### A. Compression Methods

Three methods evaluated at three compression levels (keep 75%, 50%, 25% of original):

1. **Extractive (nomic-embed-text):** Embed each sentence and query, keep top-k sentences
   by cosine similarity. Zero extra model parameters. Latency: embedding overhead only.

2. **Abstractive (qwen2:1.5b):** Summarize retrieved context to target word count.
   Higher quality compression but adds qwen2:1.5b inference time.

3. **Token budget:** Hard truncation to sentence boundary at N-word budget. Zero latency.
   Baseline for comparison.

### B. Quality Measurement

| Metric | Description |
|---|---|
| ROUGE-1 F1 | Unigram overlap between compressed and original answers |
| Answer length ratio | Compressed / original answer word count |
| Entity recall | Fraction of key entities from original answer present in compressed |

### C. Compression vs Wall Time Trade-off

_[PLACEHOLDER — fill with measured results]_

| Method | Level | Compression ratio | Compression latency (ms) | Answer ROUGE-1 | Total wall (s) |
|---|---|---|---|---|---|
| Token budget | 25% | 0.25 | ~0 | [result] | [result] |
| Token budget | 50% | 0.50 | ~0 | [result] | [result] |
| Token budget | 75% | 0.75 | ~0 | [result] | [result] |
| Extractive | 25% | ~0.25 | ~500 | [result] | [result] |
| Extractive | 50% | ~0.50 | ~500 | [result] | [result] |
| Extractive | 75% | ~0.75 | ~500 | [result] | [result] |
| Abstractive | 25% | ~0.25 | ~15000 | [result] | [result] |
| Abstractive | 50% | ~0.50 | ~8000 | [result] | [result] |
| Abstractive | 75% | ~0.75 | ~4000 | [result] | [result] |

---

## VI. Zero-Config Deployment

### A. Hardware-Aware Model Selection

`auto_config.py` detects GPU count and total VRAM, then selects model and parameters:

| Total VRAM | Selected model | ngl | Expected decode |
|---|---|---|---|
| 0 (CPU-only) | phi3:mini Q4 | 0 | 6.5 tok/s |
| <2 GB | qwen2:0.5b Q4 | 99 | ~5 tok/s est. |
| 2–8 GB | phi3:mini Q4 | 99 | 8-9 tok/s |
| 8–16 GB | qwen2.5-7B Q2_K | 99 | ~4 tok/s |
| >16 GB | qwen2.5-7B Q4 | 99 | ~6 tok/s est. |

N-gram speculative decoding (`--spec-type ngram-simple`) is enabled by default when
any GPU is detected, adding the +9.7% free improvement from v2.

### B. One-Script Installer

`install.sh` performs:
1. OS detection (Ubuntu 20.04/22.04/24.04)
2. Hardware detection via `lscpu` and `nvidia-smi`
3. Ollama installation (if absent) and model pull
4. llama-server startup with optimal settings
5. Web UI startup on port 7860
6. Performance expectations printed to console

### C. Web Interface

Minimal FastAPI application with plain HTML + SSE streaming. Features:
- Text input for queries
- Document upload (PDF, txt)
- Real-time token streaming with tok/s and VRAM display
- Mobile-responsive, no JavaScript framework dependency
- Works in any browser without installation

---

## VII. Evaluation

_[PLACEHOLDER — fill after all experiments complete]_

### A. Phase Splitting Evaluation

**Hypothesis:** CPU prefill + GPU decode reduces end-to-end latency for medium/long prompts
by routing each phase to the faster device.

**Null hypothesis:** KV cache format incompatibility prevents actual phase splitting;
theoretical speedup is achievable only with llama.cpp backend modifications.

**Metrics:** prefill_tok_s, decode_tok_s, handoff_latency_ms, total_wall_s, speedup_vs_v2_best

### B. Prompt Compression Evaluation

**Hypothesis:** Extractive compression at 50% retention reduces wall time by >40% with
<15% degradation in answer quality (ROUGE-1 F1 ≥ 0.85).

**Metrics:** compression_ratio, compression_latency_ms, rouge1_f1, entity_recall, total_wall_s

### C. Comparison Table

_[PLACEHOLDER — fill after experiments A and B complete]_

| System | Config | Mean tok/s | Mean wall (s) | Quality (ROUGE) | Notes |
|---|---|---|---|---|---|
| LegacyRAG v1 | Single GPU, Ollama | 0.95 | 469.4 | — | v1 baseline |
| LegacyRAG v2 | Dual GPU + ngram | 9.08 | 81.4 | — | v2 best |
| PhaseRAG | Phase split (theoretical) | — | [result] | — | upper bound |
| PhaseRAG | Phase split (actual) | — | [result] | — | measured |
| PhaseRAG | Extractive 50% | — | [result] | [result] | compressed |

---

## VIII. Discussion

### A. KV Cache Handoff Feasibility

_[PLACEHOLDER — fill based on Experiment A slot handoff result]_

The key engineering challenge in PhaseRAG is KV cache transfer between CPU and GPU
llama.cpp backends. The llama.cpp slot save/restore API (introduced in build b9297) is
designed for same-backend continuations. Cross-backend (CPU FP32 ↔ GPU Vulkan) handoff
requires that: (1) the serialization format is backend-agnostic, and (2) the GPU server can
map CPU-format tensors into Vulkan device memory. This is currently [RESULT_HERE].

### B. Practical Deployment Guidance

For organizations running Maxwell/Pascal Vulkan hardware in 2025–2026:

1. Enable dual-GPU layer splitting (`-ngl 99`): free 8.7× speedup over single-GPU.
2. Enable n-gram speculative decoding: free +9.7% decode speedup.
3. Use phi3-mini Q4 for interactive use (<8GB VRAM); 7B Q2_K for batch/quality workloads.
4. If phase splitting is implemented: route long prompts (>200 tokens) through CPU prefill.
5. Use prompt compression (extractive, 50%) to reduce prefill cost for RAG workloads.

---

## IX. Conclusion

_[PLACEHOLDER — fill after all experiments complete]_

PhaseRAG demonstrates that CPU-GPU heterogeneous phase splitting is [feasible / theoretically
motivated but blocked by KV cache format constraints] on legacy Maxwell Vulkan hardware.
The [X]× theoretical speedup from routing prefill to CPU directly addresses the 84–93%
prefill dominance observed in LegacyRAG v2. Prompt compression with extractive sentence
selection provides an additional [Y]% reduction in prefill cost at [Z] ROUGE-1 quality.
Together, these techniques reduce mean wall time from [v2_best: 81.4s] to [v3_best: Xs]
for the government RAG workload, making sub-minute responses feasible on pre-2017 hardware.

---

## References

[1] A. Shaik, "LegacyRAG v1: Benchmarking Open-Source LLM Inference on Legacy Vulkan Hardware,"
    SSRN 6750398, May 2026.

[2] A. Shaik, "LegacyRAG v2: Speculative Decoding and Quantization on Maxwell Vulkan,"
    IC2E 2026 Demo Paper (submitted).

[3] D. Kachris et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting,"
    ISCA 2024. arXiv:2311.18677.

[4] BatOpt: Optimizing Batch Inference for Heterogeneous GPU Clusters, TCC 2024.

[5] D. Trihinas et al., "Enabling LLM Deployment on Heterogeneous Edge Infrastructures,"
    IC2E 2024.

[6] Y. Leviathan, M. Kalman, and Y. Matias, "Fast Inference from Transformers via Speculative
    Decoding," ICML 2023. arXiv:2211.17192.

[7] T. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale,"
    NeurIPS 2022. arXiv:2208.07339.

[8] M. Abdin et al., "Phi-3 Technical Report: A Highly Capable Language Model Locally on
    Your Phone," arXiv:2404.14219, 2024.

[9] Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.

[10] H. Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of LLMs,"
     EMNLP 2023. arXiv:2310.05736.

[11] llama.cpp, "Efficient LLM Inference in C/C++," https://github.com/ggml-org/llama.cpp, 2023–2025.

---

_Last updated: 2026-05-26 — scaffold complete, v2 baseline filled, v3 experiments pending_
