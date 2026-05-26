#!/usr/bin/env python3
"""
auto_config.py — Hardware Detection and Optimal Configuration

PhaseRAG Contribution 3: Zero-Config Deployment.
Detects CPU, RAM, GPU count, and VRAM, then outputs optimal llama-server settings.
"""

import json
import os
import re
import subprocess
from pathlib import Path


def detect_cpu() -> dict:
    info: dict = {}
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.splitlines():
            if "CPU(s):" in line and "NUMA" not in line and "On-line" not in line:
                m = re.search(r":\s*(\d+)", line)
                if m:
                    info["logical_cpus"] = int(m.group(1))
            if "Core(s) per socket" in line:
                m = re.search(r":\s*(\d+)", line)
                if m:
                    info["cores_per_socket"] = int(m.group(1))
            if "Model name" in line:
                info["model_name"] = line.split(":", 1)[1].strip()
            if "Socket(s)" in line:
                m = re.search(r":\s*(\d+)", line)
                if m:
                    info["sockets"] = int(m.group(1))
    except Exception as e:
        info["error"] = str(e)
    info["physical_cores"] = (info.get("cores_per_socket", 1) *
                               info.get("sockets", 1))
    return info


def detect_ram_gb() -> float:
    try:
        out = subprocess.check_output(["free", "-b"], text=True)
        for line in out.splitlines():
            if line.startswith("Mem:"):
                total_bytes = int(line.split()[1])
                return round(total_bytes / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def detect_gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_total_mb": float(parts[2]),
                    "vram_free_mb": float(parts[3]),
                    "driver": parts[4] if len(parts) > 4 else "unknown",
                })
        return gpus
    except Exception:
        return []


def select_model(total_vram_mb: float, ram_gb: float) -> dict:
    """
    Select model based on total VRAM across all GPUs.
    Model sizes (approximate GGUF):
      phi3-mini Q4_K_M:    ~2.1 GB
      qwen2:1.5b Q4_K_M:   ~0.9 GB
      qwen2.5-7B Q2_K:     ~2.9 GB
      qwen2.5-7B Q4_K_M:   ~4.7 GB
    Reserve ~1 GB for KV cache and embeddings.
    """
    usable_vram_mb = total_vram_mb - 1024  # 1 GB reserve
    if total_vram_mb == 0:
        return {
            "model": "phi3:mini",
            "quantization": "Q4_K_M",
            "ngl": 0,
            "note": "CPU-only (no GPU detected)",
        }
    elif usable_vram_mb < 2000:
        return {
            "model": "qwen2:0.5b",
            "quantization": "Q4_K_M",
            "ngl": 99,
            "note": f"<2GB usable VRAM — smallest available model",
        }
    elif usable_vram_mb < 4000:
        return {
            "model": "phi3:mini",
            "quantization": "Q4_K_M",
            "ngl": 99,
            "note": f"4-8GB usable VRAM — phi3-mini Q4 (best quality/speed on K4200)",
        }
    elif usable_vram_mb < 8000:
        return {
            "model": "qwen2.5:7b-instruct-q2_K",
            "quantization": "Q2_K",
            "ngl": 99,
            "note": f"8-16GB usable VRAM — 7B Q2_K (larger model, slower but higher quality)",
        }
    else:
        return {
            "model": "qwen2.5:7b-instruct-q4_K_M",
            "quantization": "Q4_K_M",
            "ngl": 99,
            "note": f">16GB VRAM — 7B Q4 (best quality)",
        }


def recommend_settings(cpu: dict, ram_gb: float, gpus: list[dict]) -> dict:
    """
    Output optimal llama-server settings for detected hardware.
    Rules (tuned from LegacyRAG v2 benchmarks):
      - threads: physical cores (not hyperthreads) for CPU work
      - ngl: 99 if any GPU detected, 0 otherwise
      - ctx_size: scale with RAM, cap at 4096 for VRAM-constrained configs
      - parallel: 1 (single user, maximizes throughput)
      - ngram: always enable if GPU present (free +9.7% from v2)
    """
    physical_cores = cpu.get("physical_cores", 4)
    total_vram_mb = sum(g["vram_total_mb"] for g in gpus)
    gpu_count = len(gpus)

    ngl = 99 if gpu_count > 0 else 0
    threads = min(physical_cores, 8)
    ctx_size = 4096 if total_vram_mb >= 6000 else 2048
    use_ngram = gpu_count > 0

    settings = {
        "ngl": ngl,
        "threads": threads,
        "ctx_size": ctx_size,
        "parallel": 1,
        "ngram_speculative": use_ngram,
        "ngram_extra_args": ["--spec-type", "ngram-simple", "--spec-draft-n-max", "8"]
        if use_ngram else [],
    }
    return settings


def generate_config(output_path: Path | None = None) -> dict:
    cpu = detect_cpu()
    ram_gb = detect_ram_gb()
    gpus = detect_gpus()
    total_vram_mb = sum(g["vram_total_mb"] for g in gpus)

    model_rec = select_model(total_vram_mb, ram_gb)
    settings = recommend_settings(cpu, ram_gb, gpus)

    config = {
        "hardware": {
            "cpu": cpu,
            "ram_gb": ram_gb,
            "gpus": gpus,
            "total_vram_mb": total_vram_mb,
        },
        "recommended_model": model_rec,
        "recommended_settings": settings,
        "estimated_performance": {
            "note": "Based on LegacyRAG v2 benchmarks on dual K4200 (8GB total VRAM)",
            "phi3_mini_q4_dual_k4200_decode_tok_s": 8.28,
            "phi3_mini_q4_dual_k4200_plus_ngram_tok_s": 9.08,
            "qwen2_5_7b_q2k_dual_k4200_decode_tok_s": 3.82,
        },
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config written to {output_path}")

    return config


if __name__ == "__main__":
    cfg = generate_config(Path(__file__).parent / "legacyrag_config.json")
    print(json.dumps(cfg, indent=2))
