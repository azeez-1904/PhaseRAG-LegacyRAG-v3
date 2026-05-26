#!/usr/bin/env python3
"""
phase_splitter.py — CPU-GPU Heterogeneous Phase Splitting

PhaseRAG Contribution 1 (MLSys 2027).

Key insight from LegacyRAG v2:
  - Maxwell Vulkan GPU prefill: ~0.7 tok/s
  - CPU prefill:               ~21  tok/s  (30× faster)
  - CPU decode:                ~6.5 tok/s
  - Maxwell Vulkan GPU decode: ~8-9 tok/s  (30% faster)

Strategy: route each phase to the faster device.

Approach A — KV Cache Handoff (attempted):
  1. CPU server (-ngl 0) prefills prompt, saves slot via --slot-save-path + /slots API
  2. GPU server (-ngl 99) restores slot, decodes
  Failure mode: CPU and GPU KV cache formats may be incompatible across llama.cpp backends.
  This is documented as a research finding regardless of outcome.

Approach B — Theoretical Measurement (fallback if A fails):
  1. CPU server: full run, extract prefill_ms and decode_ms from timings
  2. GPU server: full run, extract prefill_ms and decode_ms from timings
  3. Report: theoretical_combined = cpu_prefill_ms + gpu_decode_ms
  Valid contribution: quantifies the potential gain from heterogeneous scheduling.
"""

import json
import os
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "build_b9297"
LLAMA_SERVER = BIN_DIR / "llama-server"
LIB_PATH = str(BIN_DIR)

SERVER_PORT = 8081  # use 8081 to avoid conflict with any existing server
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
SERVER_TIMEOUT = 120
SLOT_SAVE_DIR = Path("/tmp/legacyrag_v3_slots")

BLOBS_DIR = Path("/usr/share/ollama/.ollama/models/blobs")
MAX_TOKENS = 200


def find_model_gguf(manifest_path: str) -> Path:
    manifest = Path(
        "/usr/share/ollama/.ollama/models/manifests"
        f"/registry.ollama.ai/library/{manifest_path}"
    )
    with open(manifest) as f:
        data = json.load(f)
    for layer in data["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace("sha256:", "sha256-")
            return BLOBS_DIR / digest
    raise FileNotFoundError(f"No model layer in manifest: {manifest_path}")


def get_vram() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        result = []
        for line in out.strip().splitlines():
            idx, free, used, total = [x.strip() for x in line.split(",")]
            result.append({"gpu": int(idx), "free_mb": float(free),
                           "used_mb": float(used), "total_mb": float(total)})
        return result
    except Exception as e:
        return [{"error": str(e)}]


def _start_server(model_path: Path, ngl: int, extra_args: list[str] | None = None,
                  log_tag: str = "") -> subprocess.Popen:
    SLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "-ngl", str(ngl),
        "--port", str(SERVER_PORT),
        "--host", "127.0.0.1",
        "--ctx-size", "2048",
        "--threads", "4",
        "--parallel", "1",
        "--slots",
        "--slot-save-path", str(SLOT_SAVE_DIR),
        "--log-disable",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log_path = Path(__file__).parent / "results" / "phase_server.log"
    log_path.parent.mkdir(exist_ok=True)
    log_file = open(log_path, "a")
    tag = log_tag or f"ngl={ngl}"
    log_file.write(f"\n\n=== {tag} {datetime.now(timezone.utc).isoformat()} ===\n")
    log_file.write(f"CMD: {' '.join(cmd)}\n")
    log_file.flush()

    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    print(f"    Server PID {proc.pid} (ngl={ngl}), waiting for health...", flush=True)

    deadline = time.time() + SERVER_TIMEOUT
    while time.time() < deadline:
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
                if r.status == 200:
                    print(f"    Server healthy (ngl={ngl}).", flush=True)
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            log_file.close()
            raise RuntimeError(
                f"llama-server (ngl={ngl}) exited early (code {proc.returncode}). "
                f"Check {log_path}"
            )

    proc.terminate()
    log_file.close()
    raise TimeoutError(f"Server did not become healthy within {SERVER_TIMEOUT}s")


def _stop_server(proc: subprocess.Popen, sleep_s: int = 30) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print(f"    Server stopped. Sleeping {sleep_s}s for VRAM/memory to clear...", flush=True)
    time.sleep(sleep_s)


def _post_json(url: str, payload: dict, timeout: int = 900) -> tuple[dict, float]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    elapsed = time.perf_counter() - t0
    return json.loads(raw), elapsed


def _extract_timings(resp: dict, wall_s: float) -> dict:
    t = resp.get("timings", {})
    prefill_n = t.get("prompt_n", 0)
    prefill_ms = t.get("prompt_ms", 0)
    decode_n = t.get("predicted_n", 0)
    decode_ms = t.get("predicted_ms", 0)
    prefill_tps = t.get("prompt_per_second", None)
    decode_tps = t.get("predicted_per_second", None)
    if prefill_tps is None and prefill_ms > 0:
        prefill_tps = prefill_n / (prefill_ms / 1000)
    if decode_tps is None and decode_ms > 0:
        decode_tps = decode_n / (decode_ms / 1000)
    return {
        "prefill_n": prefill_n,
        "prefill_ms": round(prefill_ms, 2),
        "prefill_tok_s": round(prefill_tps, 4) if prefill_tps else None,
        "decode_n": decode_n,
        "decode_ms": round(decode_ms, 2),
        "decode_tok_s": round(decode_tps, 4) if decode_tps else None,
        "total_wall_s": round(wall_s, 3),
    }


def run_cpu_only(model_path: Path, prompt: str) -> dict:
    """Full inference on CPU (-ngl 0). Returns timing breakdown."""
    print("  [CPU-only] Starting CPU server...", flush=True)
    proc = _start_server(model_path, ngl=0, log_tag="CPU-only")
    time.sleep(2)
    try:
        resp, wall_s = _post_json(
            f"{SERVER_URL}/completion",
            {"prompt": prompt, "n_predict": MAX_TOKENS,
             "temperature": 0.1, "stop": ["</s>", "<|end|>"], "stream": False}
        )
        result = _extract_timings(resp, wall_s)
        result["mode"] = "cpu_only"
        result["ngl"] = 0
        print(
            f"  [CPU-only] prefill={result['prefill_tok_s']:.2f} tok/s  "
            f"decode={result['decode_tok_s']:.2f} tok/s  wall={wall_s:.1f}s",
            flush=True,
        )
        return result
    finally:
        _stop_server(proc, sleep_s=15)


def run_gpu_only(model_path: Path, prompt: str) -> dict:
    """Full inference on GPU (-ngl 99). Returns timing breakdown."""
    print("  [GPU-only] Starting GPU server...", flush=True)
    proc = _start_server(model_path, ngl=99, log_tag="GPU-only")
    time.sleep(2)
    try:
        resp, wall_s = _post_json(
            f"{SERVER_URL}/completion",
            {"prompt": prompt, "n_predict": MAX_TOKENS,
             "temperature": 0.1, "stop": ["</s>", "<|end|>"], "stream": False}
        )
        result = _extract_timings(resp, wall_s)
        result["mode"] = "gpu_only"
        result["ngl"] = 99
        print(
            f"  [GPU-only] prefill={result['prefill_tok_s']:.2f} tok/s  "
            f"decode={result['decode_tok_s']:.2f} tok/s  wall={wall_s:.1f}s",
            flush=True,
        )
        return result
    finally:
        _stop_server(proc, sleep_s=15)


def run_gpu_ngram(model_path: Path, prompt: str) -> dict:
    """GPU + ngram-simple speculative decoding (v2 best config)."""
    print("  [GPU+ngram] Starting GPU+ngram server...", flush=True)
    proc = _start_server(
        model_path, ngl=99,
        extra_args=["--spec-type", "ngram-simple", "--spec-draft-n-max", "8"],
        log_tag="GPU+ngram"
    )
    time.sleep(2)
    try:
        resp, wall_s = _post_json(
            f"{SERVER_URL}/completion",
            {"prompt": prompt, "n_predict": MAX_TOKENS,
             "temperature": 0.1, "stop": ["</s>", "<|end|>"], "stream": False}
        )
        result = _extract_timings(resp, wall_s)
        result["mode"] = "gpu_ngram"
        result["ngl"] = 99
        result["ngram"] = True
        print(
            f"  [GPU+ngram] prefill={result['prefill_tok_s']:.2f} tok/s  "
            f"decode={result['decode_tok_s']:.2f} tok/s  wall={wall_s:.1f}s",
            flush=True,
        )
        return result
    finally:
        _stop_server(proc, sleep_s=15)


def attempt_slot_handoff(model_path: Path, prompt: str) -> dict:
    """
    Attempt A: CPU prefill → save slot → GPU restore → decode.

    Returns a result dict with mode="phase_split_actual" if successful,
    or mode="phase_split_failed" with failure_reason if the handoff fails.
    """
    slot_file = "slot0_handoff.bin"
    slot_path = SLOT_SAVE_DIR / slot_file
    if slot_path.exists():
        slot_path.unlink()

    # ── Step 1: CPU server, prefill only (n_predict=0) ────────────────────────
    print("  [Phase-split] Step 1: CPU prefill...", flush=True)
    cpu_proc = _start_server(model_path, ngl=0, log_tag="PhaseA-CPU-prefill")
    time.sleep(2)

    prefill_timings = None
    slot_saved = False
    failure_reason = None

    try:
        # Send prefill-only request
        t0 = time.perf_counter()
        resp, wall_s = _post_json(
            f"{SERVER_URL}/completion",
            {"prompt": prompt, "n_predict": 0,
             "temperature": 0.1, "stop": ["</s>", "<|end|>"],
             "stream": False, "cache_prompt": True}
        )
        prefill_elapsed = time.perf_counter() - t0
        t = resp.get("timings", {})
        prefill_n = t.get("prompt_n", 0)
        prefill_ms = t.get("prompt_ms", 0)
        prefill_tps = t.get("prompt_per_second", None)
        if prefill_tps is None and prefill_ms > 0:
            prefill_tps = prefill_n / (prefill_ms / 1000)
        prefill_timings = {
            "prefill_n": prefill_n,
            "prefill_ms": round(prefill_ms, 2),
            "prefill_tok_s": round(prefill_tps, 4) if prefill_tps else None,
            "prefill_wall_s": round(prefill_elapsed, 3),
        }
        print(f"    CPU prefill done: {prefill_tps:.2f} tok/s, {prefill_ms:.0f}ms", flush=True)

        # ── Step 2: Save slot state ────────────────────────────────────────────
        print("  [Phase-split] Step 2: Saving slot state...", flush=True)
        try:
            save_payload = {"action": "save", "filename": slot_file}
            save_resp, _ = _post_json(f"{SERVER_URL}/slots/0", save_payload, timeout=30)
            if slot_path.exists() and slot_path.stat().st_size > 0:
                slot_saved = True
                print(f"    Slot saved: {slot_path} ({slot_path.stat().st_size / 1024:.1f} KB)", flush=True)
            else:
                failure_reason = (
                    f"Slot save API returned {save_resp} but file not found or empty at {slot_path}"
                )
                print(f"    WARNING: {failure_reason}", flush=True)
        except Exception as e:
            failure_reason = f"Slot save API error: {type(e).__name__}: {e}"
            print(f"    WARNING: {failure_reason}", flush=True)

    except Exception as e:
        failure_reason = f"CPU prefill failed: {type(e).__name__}: {e}"
        print(f"    ERROR: {failure_reason}", flush=True)
    finally:
        _stop_server(cpu_proc, sleep_s=20)

    if not slot_saved or failure_reason:
        return {
            "mode": "phase_split_failed",
            "failure_reason": failure_reason or "slot not saved",
            "prefill_timings": prefill_timings,
        }

    # ── Step 3: GPU server, restore slot and decode ───────────────────────────
    print("  [Phase-split] Step 3: GPU server, restoring slot and decoding...", flush=True)
    gpu_proc = _start_server(model_path, ngl=99, log_tag="PhaseA-GPU-decode")
    time.sleep(2)

    try:
        # Restore slot
        t_restore_start = time.perf_counter()
        try:
            restore_payload = {"action": "restore", "filename": slot_file}
            restore_resp, _ = _post_json(f"{SERVER_URL}/slots/0", restore_payload, timeout=30)
            handoff_ms = (time.perf_counter() - t_restore_start) * 1000
            print(f"    Slot restored in {handoff_ms:.1f}ms. Response: {restore_resp}", flush=True)
        except Exception as e:
            failure_reason = f"Slot restore API error: {type(e).__name__}: {e}"
            print(f"    ERROR: {failure_reason}", flush=True)
            return {
                "mode": "phase_split_failed",
                "failure_reason": failure_reason,
                "prefill_timings": prefill_timings,
            }

        # Decode: send empty prompt; GPU should continue from restored KV state
        t_decode_start = time.perf_counter()
        try:
            resp, decode_wall_s = _post_json(
                f"{SERVER_URL}/completion",
                {"prompt": "", "n_predict": MAX_TOKENS,
                 "temperature": 0.1, "stop": ["</s>", "<|end|>"], "stream": False}
            )
            decode_elapsed = time.perf_counter() - t_decode_start
            dt = resp.get("timings", {})
            decode_n = dt.get("predicted_n", 0)
            decode_ms = dt.get("predicted_ms", 0)
            decode_tps = dt.get("predicted_per_second", None)
            if decode_tps is None and decode_ms > 0 and decode_n > 0:
                decode_tps = decode_n / (decode_ms / 1000)

            total_wall_s = prefill_timings["prefill_wall_s"] + handoff_ms / 1000 + decode_wall_s

            result = {
                "mode": "phase_split_actual",
                "prefill_n": prefill_timings["prefill_n"],
                "prefill_ms": prefill_timings["prefill_ms"],
                "prefill_tok_s": prefill_timings["prefill_tok_s"],
                "prefill_wall_s": prefill_timings["prefill_wall_s"],
                "handoff_ms": round(handoff_ms, 2),
                "decode_n": decode_n,
                "decode_ms": round(decode_ms, 2),
                "decode_tok_s": round(decode_tps, 4) if decode_tps else None,
                "decode_wall_s": round(decode_elapsed, 3),
                "total_wall_s": round(total_wall_s, 3),
                "slot_file_size_kb": round(slot_path.stat().st_size / 1024, 1),
            }
            print(
                f"  [Phase-split] ACTUAL split: prefill={result['prefill_tok_s']:.2f} tok/s (CPU), "
                f"decode={result['decode_tok_s']:.2f} tok/s (GPU), "
                f"handoff={handoff_ms:.1f}ms, total={total_wall_s:.1f}s",
                flush=True,
            )
            return result

        except Exception as e:
            failure_reason = f"GPU decode after restore failed: {type(e).__name__}: {e}"
            print(f"    ERROR: {failure_reason}", flush=True)
            return {
                "mode": "phase_split_failed",
                "failure_reason": failure_reason,
                "prefill_timings": prefill_timings,
            }
    finally:
        _stop_server(gpu_proc, sleep_s=20)


def compute_theoretical_phase_split(cpu_result: dict, gpu_result: dict) -> dict:
    """
    Compute theoretical phase split from separate CPU and GPU measurements.
    theoretical_total = cpu_prefill_time + gpu_decode_time
    This is valid even if KV cache handoff fails: quantifies the potential speedup.
    """
    cpu_prefill_ms = cpu_result.get("prefill_ms", 0)
    cpu_decode_ms = cpu_result.get("decode_ms", 0)
    gpu_prefill_ms = gpu_result.get("prefill_ms", 0)
    gpu_decode_ms = gpu_result.get("decode_ms", 0)

    theoretical_total_ms = cpu_prefill_ms + gpu_decode_ms
    cpu_only_total_ms = cpu_prefill_ms + cpu_decode_ms
    gpu_only_total_ms = gpu_prefill_ms + gpu_decode_ms

    speedup_vs_cpu = cpu_only_total_ms / theoretical_total_ms if theoretical_total_ms > 0 else None
    speedup_vs_gpu = gpu_only_total_ms / theoretical_total_ms if theoretical_total_ms > 0 else None

    prefill_pct_in_gpu = (gpu_prefill_ms / gpu_only_total_ms * 100) if gpu_only_total_ms > 0 else None

    return {
        "mode": "phase_split_theoretical",
        "cpu_prefill_ms": round(cpu_prefill_ms, 2),
        "cpu_prefill_tok_s": cpu_result.get("prefill_tok_s"),
        "gpu_decode_ms": round(gpu_decode_ms, 2),
        "gpu_decode_tok_s": gpu_result.get("decode_tok_s"),
        "theoretical_total_ms": round(theoretical_total_ms, 2),
        "theoretical_total_s": round(theoretical_total_ms / 1000, 3),
        "cpu_only_total_ms": round(cpu_only_total_ms, 2),
        "gpu_only_total_ms": round(gpu_only_total_ms, 2),
        "speedup_vs_cpu_only": round(speedup_vs_cpu, 3) if speedup_vs_cpu else None,
        "speedup_vs_gpu_only": round(speedup_vs_gpu, 3) if speedup_vs_gpu else None,
        "gpu_prefill_pct_of_gpu_total": round(prefill_pct_in_gpu, 1) if prefill_pct_in_gpu else None,
    }


if __name__ == "__main__":
    model = find_model_gguf("phi3/mini")
    prompt = (
        "Explain how retrieval-augmented generation works and why it is useful for "
        "enterprise applications. Describe the role of embedding models and vector stores "
        "in a RAG pipeline. How is retrieved context injected into the prompt, and what "
        "are the main failure modes when the retrieved chunks are irrelevant or too long? "
        "Provide a concrete example with a government records use case."
    )
    print(f"Model: {model}\n")
    print("=== Single-prompt phase split test ===\n")

    cpu = run_cpu_only(model, prompt)
    print(f"\nCPU result: {cpu}\n")

    gpu = run_gpu_only(model, prompt)
    print(f"\nGPU result: {gpu}\n")

    theoretical = compute_theoretical_phase_split(cpu, gpu)
    print(f"\nTheoretical phase split: {theoretical}\n")

    print("\n=== Attempting KV cache slot handoff ===\n")
    handoff = attempt_slot_handoff(model, prompt)
    print(f"\nHandoff result: {handoff}")
