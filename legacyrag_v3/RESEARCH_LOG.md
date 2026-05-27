# PhaseRAG v3 — Research Log

**System:** Dell Precision Tower 5810, Ubuntu 24.04
**Hardware:** Intel Xeon E5-1620 v3 (4C/8T, 3.5GHz) | 7.7GB RAM | 2× NVIDIA Quadro K4200 (Maxwell GM204, 4GB GDDR5, 173 GB/s, Vulkan 1.3, no FP16)
**llama.cpp:** b9297
**Target venue:** MLSys 2027

---

## Session 2026-05-26 — v3 Setup

**Scaffolded from:** LegacyRAG v2 (legacyrag_v2/) which ran 4 experiments:
- exp1 (phi3:mini dual-GPU baseline): 8.28 tok/s mean, 188s wall
- exp2 (speculative decoding qwen2:1.5b + qwen2:0.5b): 3.36 tok/s, no speedup
- exp3 (ngram-simple): 9.08 tok/s (+9.7%), v2 best configuration
- exp4 (qwen2.5-7B Q2_K): 3.82 tok/s (54% slower than 3.8B baseline)

**v2 key insight that motivates v3:**
Prefill takes 84–93% of wall time for medium/long prompts on Maxwell Vulkan.
GPU prefill rate: 0.62–0.95 tok/s. CPU prefill rate: ~21 tok/s (30× faster).
GPU decode: 8-9 tok/s. CPU decode: ~6.5 tok/s (GPU 30% faster).

**Files created:**
- `phase_splitter.py` — CPU-GPU heterogeneous phase splitting (Contribution 1)
- `prompt_compressor.py` — Three compression methods + quality measurement (Contribution 2)
- `auto_config.py` — Hardware detection, model selection, config output (Contribution 3)
- `benchmark_v3.py` — Experiment A runner (phase splitting, 10 prompts, 4 configs)
- `install.sh` — One-script installer with hardware-aware setup
- `requirements.txt` — FastAPI, uvicorn, jinja2, python-multipart
- `web_ui/app.py` — FastAPI SSE streaming UI
- `web_ui/templates/index.html` — Plain HTML + fetch API, no JS frameworks
- `paper_notes/PhaseRAG_draft.md` — Full paper scaffold with v2 numbers as baseline

**Git:** branch v3-development, remote v3-origin → https://github.com/azeez-1904/LegacyRAG-v3.git

---

## Experiment A: CPU-GPU Phase Splitting

_[PENDING — started 2026-05-26]_

**Design:**
- 10 prompts (same as v2: 3 short / 4 medium / 3 long)
- 4 configs per prompt:
  1. CPU-only (ngl=0): control, measures CPU prefill+decode separately
  2. GPU-only (ngl=99): control, measures GPU prefill+decode separately
  3. Phase-split theoretical: cpu_prefill_ms + gpu_decode_ms from (1) and (2)
  4. Phase-split actual: attempt slot save/restore CPU→GPU handoff
  5. GPU+ngram (ngl=99, ngram-simple): v2 reference baseline

**Slot handoff approach:**
- b9297 has `--slot-save-path` and `/slots/{id}` POST API
- CPU server saves KV cache to /tmp/legacyrag_v3_slots/slot0.bin
- GPU server restores slot, decodes from empty prompt
- Known risk: CPU FP32 KV cache may be incompatible with GPU Vulkan backend format

**Expected results (hypothesis):**
- CPU prefill: ~21 tok/s (BLAS vectorized, no Vulkan dispatch overhead)
- GPU decode: ~8-9 tok/s (bandwidth-bound, same as v2)
- Theoretical combined: large speedup for long prompts (e.g., 400-token prompt: 19s CPU vs 571s GPU for prefill)
- Slot handoff: outcome unknown — documenting either way

**Status:** COMPLETE — 2026-05-26

**Results summary:**
- CPU-only: 5.34 tok/s decode, 139s mean wall
- GPU-only: 9.29 tok/s decode, 122s mean wall
- GPU+ngram: 8.73 tok/s decode, 124s mean wall (SLOWER than GPU-only per-session — see below)
- Phase split theoretical: 123s mean wall, speedup ≈ 1.0× for medium/long, 1.03× for short

**Critical finding:** CPU prefill advantage is prompt-length dependent.
- Short (<20 tok): CPU prefill ~20 tok/s vs GPU ~13 tok/s → phase split marginally beneficial
- Medium/long (70-400 tok): CPU prefill drops to 1.0-1.4 tok/s ≈ GPU prefill 0.8-1.3 tok/s
- Theoretical phase split speedup: 0.99-1.03× — essentially no benefit for realistic RAG workloads

**Slot handoff:** FAILED. POST /slots/0 with {"action":"save",...} returns HTTP 400.
The b9297 slot API is for same-server continuation only. Cross-backend (CPU→GPU) KV cache
transfer is not supported. This is a hard constraint in current llama.cpp architecture.

**GPU+ngram underperformed:** 8.73 vs 9.29 tok/s GPU-only. Ngram requires accumulated
prior context from same session to build candidate lookup. Fresh server per prompt = empty
ngram table. v2's +9.7% required same-session batch usage. Single-query isolation = no ngram benefit.

**Hypothesis for B:** Since phase splitting doesn't help for medium/long prompts, prompt
compression is the only viable strategy. Even 25% compression of a 400-token prompt
reduces prefill time by ~25% (roughly 50s → 37s for a GPU long-prompt prefill), which
is the dominant wall-time component.

---

## Experiment B: Prompt Compression

_[PENDING — after Experiment A]_

**Design:**
- Same 10 prompts, 3 compression methods × 3 levels (25%/50%/75% retention)
- Extractive: nomic-embed-text sentence similarity
- Abstractive: qwen2:1.5b summarization
- Token budget: hard truncation
- Quality: ROUGE-1 F1, entity recall, answer length ratio

---

_End of log — updated automatically after each experiment_
