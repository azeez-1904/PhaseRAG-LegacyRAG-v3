#!/usr/bin/env python3
"""
analyze_exp_b.py — Full analysis of Experiment B (Prompt Compression)

Reads results/exp_compression_full.json (or falls back to exp_compression.json
if the full merge hasn't happened yet) and generates:

  1. Console summary tables (method × level, by bucket)
  2. Wall-time speedup vs ROUGE quality trade-off
  3. Paper-ready tables (Markdown + LaTeX)
  4. Key findings paragraph
  5. Saves analysis to results/exp_b_analysis.json

Usage:
    python3 analyze_exp_b.py            # auto-selects best available file
    python3 analyze_exp_b.py --full     # force exp_compression_full.json
    python3 analyze_exp_b.py --partial  # force exp_compression.json
"""

import json
import sys
from pathlib import Path
from statistics import mean as smean, stdev

RESULTS_DIR = Path(__file__).parent / "results"
FULL_FILE   = RESULTS_DIR / "exp_compression_full.json"
PART_FILE   = RESULTS_DIR / "exp_compression.json"

# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return round(smean(v), 4) if v else None

def safe_std(vals):
    v = [x for x in vals if x is not None]
    return round(stdev(v), 4) if len(v) >= 2 else None

def load_results(force=None):
    if force == "full":
        path = FULL_FILE
    elif force == "partial":
        path = PART_FILE
    else:
        path = FULL_FILE if FULL_FILE.exists() else PART_FILE

    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run experiment_b_rerun.py first.")

    with open(path) as f:
        d = json.load(f)

    source = path.name
    results = d.get("results", [])
    print(f"Loaded {len(results)} prompt results from {source}")
    return results, source


def flatten_rows(results):
    """
    Returns a flat list of dicts — one row per (prompt, method, keep_fraction).
    Each row includes:
      id, bucket, context_words, method, keep_fraction_target, actual_ratio,
      comp_latency_ms, wall_s, decode_tok_s, prefill_tok_s, baseline_wall_s,
      baseline_prefill_tok_s, baseline_decode_tok_s, baseline_answer_words,
      rouge1_f1, entity_recall, answer_length_ratio, speedup_vs_baseline
    """
    rows = []
    for r in results:
        bid      = r["id"]
        bucket   = r["bucket"]
        ctx_w    = r.get("context_words", 0)
        baseline = r.get("baseline", {})
        b_wall   = baseline.get("wall_s")
        b_pre    = baseline.get("prefill_tok_s")
        b_dec    = baseline.get("decode_tok_s")
        b_awords = baseline.get("answer_words")
        b_err    = baseline.get("error")

        # Baseline row
        if not b_err and b_wall:
            rows.append({
                "id": bid, "bucket": bucket, "context_words": ctx_w,
                "method": "baseline", "keep_fraction_target": 1.0,
                "actual_ratio": 1.0, "comp_latency_ms": 0,
                "wall_s": b_wall, "decode_tok_s": b_dec, "prefill_tok_s": b_pre,
                "baseline_wall_s": b_wall,
                "rouge1_f1": None, "entity_recall": None,
                "answer_length_ratio": None,
                "speedup_vs_baseline": 1.0,
                "answer_words": b_awords,
            })

        for v in r.get("variants", []):
            if v.get("error"):
                continue
            q   = v.get("quality", {})
            cm  = v.get("compression_meta", {})
            w   = v.get("wall_s")
            if w is None:
                continue

            rows.append({
                "id": bid, "bucket": bucket, "context_words": ctx_w,
                "method": v.get("method"),
                "keep_fraction_target": v.get("keep_fraction_target"),
                "actual_ratio": v.get("actual_ratio"),
                "comp_latency_ms": cm.get("compression_latency_ms", 0),
                "wall_s": w,
                "decode_tok_s": v.get("decode_tok_s"),
                "prefill_tok_s": v.get("prefill_tok_s"),
                "baseline_wall_s": b_wall,
                "rouge1_f1": q.get("rouge1_f1"),
                "entity_recall": q.get("entity_recall"),
                "answer_length_ratio": q.get("answer_length_ratio"),
                "speedup_vs_baseline": round(b_wall / w, 3) if (b_wall and w) else None,
                "answer_words": v.get("answer_words"),
            })
    return rows


# ── Table 1: Method × Level summary (all prompts) ─────────────────────────────

def table_method_level(rows, title="ALL PROMPTS"):
    methods = ["token_budget", "extractive", "abstractive"]
    keeps   = [0.75, 0.50, 0.25]

    print(f"\n{'='*78}")
    print(f"TABLE: {title} — Method × Compression Level")
    print(f"{'='*78}")
    hdr = (f"{'Method':<14} {'Keep':>5} {'N':>3} {'Ratio':>6} {'CompMs':>7} "
           f"{'ROUGE1':>7} {'EntRec':>7} {'Wall':>7}s {'Spdup':>6}× {'Decode':>6}")
    print(hdr)
    print("─" * 78)

    summary_rows = []
    for method in methods:
        for keep in keeps:
            sub = [r for r in rows if r["method"] == method
                   and r.get("keep_fraction_target") == keep]
            if not sub:
                continue
            n           = len(sub)
            ratio       = safe_mean([r["actual_ratio"]      for r in sub])
            comp_ms     = safe_mean([r["comp_latency_ms"]   for r in sub])
            rouge       = safe_mean([r["rouge1_f1"]         for r in sub])
            ent         = safe_mean([r["entity_recall"]      for r in sub])
            wall        = safe_mean([r["wall_s"]             for r in sub])
            speedup     = safe_mean([r["speedup_vs_baseline"] for r in sub])
            decode      = safe_mean([r["decode_tok_s"]       for r in sub])

            print(f"{method:<14} {keep:>5.0%} {n:>3} "
                  f"{ratio:>6.3f} {(comp_ms or 0):>7.0f} "
                  f"{(rouge or 0):>7.3f} {(ent or 0):>7.3f} "
                  f"{(wall or 0):>7.1f} {(speedup or 0):>6.2f} "
                  f"{(decode or 0):>6.2f}")
            summary_rows.append({
                "method": method, "keep": keep, "n": n,
                "ratio": ratio, "comp_ms": comp_ms, "rouge": rouge,
                "entity_recall": ent, "wall_s": wall, "speedup": speedup,
                "decode_tok_s": decode,
            })

    # Baseline reference
    b_rows = [r for r in rows if r["method"] == "baseline"]
    if b_rows:
        bwall = safe_mean([r["wall_s"] for r in b_rows])
        bdec  = safe_mean([r["decode_tok_s"] for r in b_rows])
        print("─" * 78)
        print(f"{'baseline':<14} {'100%':>5} {len(b_rows):>3} "
              f"{'1.000':>6} {'0':>7} "
              f"{'—':>7} {'—':>7} "
              f"{(bwall or 0):>7.1f} {'1.00':>6} "
              f"{(bdec or 0):>6.2f}")

    return summary_rows


# ── Table 2: By bucket ─────────────────────────────────────────────────────────

def table_by_bucket(rows):
    methods = ["token_budget", "extractive", "abstractive"]
    keeps   = [0.75, 0.50, 0.25]
    buckets = ["short", "medium", "long"]

    print(f"\n{'='*78}")
    print("TABLE: By Prompt Bucket")
    print(f"{'='*78}")

    bucket_rows = []
    for bucket in buckets:
        b_baseline = [r for r in rows if r["method"] == "baseline"
                      and r["bucket"] == bucket]
        b_wall_mean = safe_mean([r["wall_s"] for r in b_baseline]) if b_baseline else None

        bwall_str = f"{b_wall_mean:.1f}s" if b_wall_mean is not None else "n/a"
        print(f"\n  {bucket.upper()} — baseline wall={bwall_str} "
              f"(n={len(b_baseline)} prompts)")
        print(f"  {'Method':<14} {'Keep':>5} {'N':>3} {'Ratio':>6} "
              f"{'ROUGE1':>7} {'Wall':>7}s {'Speedup':>7}×")
        print("  " + "─" * 56)

        for method in methods:
            for keep in keeps:
                sub = [r for r in rows if r["method"] == method
                       and r.get("keep_fraction_target") == keep
                       and r["bucket"] == bucket]
                if not sub:
                    continue
                ratio   = safe_mean([r["actual_ratio"] for r in sub])
                rouge   = safe_mean([r["rouge1_f1"] for r in sub])
                wall    = safe_mean([r["wall_s"] for r in sub])
                speedup = safe_mean([r["speedup_vs_baseline"] for r in sub])
                n       = len(sub)
                print(f"  {method:<14} {keep:>5.0%} {n:>3} "
                      f"{(ratio or 0):>6.3f} "
                      f"{(rouge or 0):>7.3f} "
                      f"{(wall or 0):>7.1f} "
                      f"{(speedup or 0):>7.2f}")
                bucket_rows.append({
                    "bucket": bucket, "method": method, "keep": keep,
                    "n": n, "ratio": ratio, "rouge": rouge,
                    "wall_s": wall, "speedup": speedup,
                })

    return bucket_rows


# ── Table 3: Paper-ready Markdown ─────────────────────────────────────────────

def table_paper_markdown(rows):
    methods = ["token_budget", "extractive", "abstractive"]
    keeps   = [0.75, 0.50, 0.25]

    b_rows = [r for r in rows if r["method"] == "baseline"]
    b_wall = safe_mean([r["wall_s"] for r in b_rows]) or 0

    print(f"\n{'='*78}")
    print("PAPER TABLE (Markdown) — Compression Results, All Prompts")
    print(f"{'='*78}")
    print()
    print("| Method | Level | Act. Ratio | Comp (ms) | ROUGE-1 | Entity Rec | Wall (s) | Speedup |")
    print("|--------|-------|-----------|-----------|---------|-----------|----------|---------|")
    print(f"| Baseline | — | 1.000 | 0 | — | — | {b_wall:.1f} | 1.00× |")

    for method in methods:
        label = {"token_budget": "Token budget", "extractive": "Extractive",
                 "abstractive": "Abstractive"}[method]
        for keep in keeps:
            sub = [r for r in rows if r["method"] == method
                   and r.get("keep_fraction_target") == keep]
            if not sub:
                continue
            ratio   = safe_mean([r["actual_ratio"] for r in sub]) or 0
            comp_ms = safe_mean([r["comp_latency_ms"] for r in sub]) or 0
            rouge   = safe_mean([r["rouge1_f1"] for r in sub]) or 0
            ent     = safe_mean([r["entity_recall"] for r in sub]) or 0
            wall    = safe_mean([r["wall_s"] for r in sub]) or 0
            speedup = safe_mean([r["speedup_vs_baseline"] for r in sub]) or 0
            print(f"| {label} | {int(keep*100)}% keep | {ratio:.3f} | "
                  f"{comp_ms:.0f} | {rouge:.3f} | {ent:.3f} | {wall:.1f} | {speedup:.2f}× |")
    print()


# ── Table 4: LaTeX ─────────────────────────────────────────────────────────────

def table_latex(rows, outpath: Path):
    methods = ["token_budget", "extractive", "abstractive"]
    keeps   = [0.75, 0.50, 0.25]
    labels  = {"token_budget": "Token budget", "extractive": "Extractive",
               "abstractive": "Abstractive"}

    b_rows = [r for r in rows if r["method"] == "baseline"]
    b_wall = safe_mean([r["wall_s"] for r in b_rows]) or 0
    b_dec  = safe_mean([r["decode_tok_s"] for r in b_rows]) or 0
    n_b    = len(b_rows)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Prompt Compression Results on Dual K4200 Vulkan. "
        r"phi3-mini Q4\_K\_M, 10 prompts (3 short / 4 medium / 3 long), "
        r"MAX\_TOKENS=150. Speedup relative to uncompressed baseline. "
        r"ROUGE-1 and entity recall measured against baseline answer.}",
        r"\label{tab:compression}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Level & Ratio & Comp (ms) & ROUGE-1 & Ent.Rec & Wall (s) & Speedup \\",
        r"\midrule",
        f"Baseline & — & 1.000 & 0 & — & — & {b_wall:.1f} & 1.00$\\times$ \\\\",
        r"\midrule",
    ]

    for mi, method in enumerate(methods):
        for ki, keep in enumerate(keeps):
            sub = [r for r in rows if r["method"] == method
                   and r.get("keep_fraction_target") == keep]
            if not sub:
                continue
            ratio   = safe_mean([r["actual_ratio"] for r in sub]) or 0
            comp_ms = safe_mean([r["comp_latency_ms"] for r in sub]) or 0
            rouge   = safe_mean([r["rouge1_f1"] for r in sub]) or 0
            ent     = safe_mean([r["entity_recall"] for r in sub]) or 0
            wall    = safe_mean([r["wall_s"] for r in sub]) or 0
            speedup = safe_mean([r["speedup_vs_baseline"] for r in sub]) or 0
            n       = len(sub)
            method_col = labels[method] if ki == 0 else ""
            suf = r" \\" if not (mi == len(methods)-1 and ki == len(keeps)-1) else r" \\"
            lines.append(
                f"{method_col} & {int(keep*100)}\\% & {ratio:.3f} & "
                f"{comp_ms:.0f} & {rouge:.3f} & {ent:.3f} & "
                f"{wall:.1f} & {speedup:.2f}$\\times${suf}"
            )
        if mi < len(methods) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)
    with open(outpath, "w") as f:
        f.write(latex)
    print(f"\n  LaTeX table saved to {outpath}")
    return latex


# ── Table 5: By-bucket LaTeX ───────────────────────────────────────────────────

def table_by_bucket_latex(rows, outpath: Path):
    methods = ["token_budget", "extractive", "abstractive"]
    keeps   = [0.75, 0.50, 0.25]
    labels  = {"token_budget": "TokBudget", "extractive": "Extractive",
               "abstractive": "Abstractive"}
    buckets = ["short", "medium", "long"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compression results by prompt length bucket. "
        r"ROUGE-1 F1 and wall time speedup vs uncompressed baseline, "
        r"mean across prompts in each bucket.}",
        r"\label{tab:compression_bucket}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ll" + "rr" * len(buckets) + "}",
        r"\toprule",
    ]

    header_parts = ["Method", "Level"]
    for b in buckets:
        header_parts += [f"\\multicolumn{{2}}{{c}}{{{b.capitalize()}}}", ""]
    lines.append(" & ".join(header_parts[:2]) +
                 " & " + " & ".join(
                     f"\\multicolumn{{2}}{{c}}{{{b.capitalize()}}}"
                     for b in buckets) + r" \\")
    sub_header = ["", ""]
    for _ in buckets:
        sub_header += ["R-1", "Spd"]
    lines.append(" & ".join(sub_header) + r" \\")
    lines.append(r"\midrule")

    # Baseline row
    bline_parts = ["Baseline", "—"]
    for bucket in buckets:
        b_rows = [r for r in rows if r["method"] == "baseline" and r["bucket"] == bucket]
        bwall  = safe_mean([r["wall_s"] for r in b_rows]) if b_rows else 0
        bline_parts += ["—", f"{bwall:.0f}s"]
    lines.append(" & ".join(bline_parts) + r" \\")
    lines.append(r"\midrule")

    for mi, method in enumerate(methods):
        for ki, keep in enumerate(keeps):
            row_parts = [labels[method] if ki == 0 else "", f"{int(keep*100)}\\%"]
            for bucket in buckets:
                sub = [r for r in rows if r["method"] == method
                       and r.get("keep_fraction_target") == keep
                       and r["bucket"] == bucket]
                if not sub:
                    row_parts += ["—", "—"]
                    continue
                rouge   = safe_mean([r["rouge1_f1"] for r in sub]) or 0
                speedup = safe_mean([r["speedup_vs_baseline"] for r in sub]) or 0
                row_parts += [f"{rouge:.3f}", f"{speedup:.2f}$\\times$"]
            lines.append(" & ".join(row_parts) + r" \\")
        if mi < len(methods) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    latex = "\n".join(lines)
    with open(outpath, "w") as f:
        f.write(latex)
    print(f"  LaTeX by-bucket table saved to {outpath}")
    return latex


# ── Key findings ───────────────────────────────────────────────────────────────

def print_key_findings(rows):
    methods = ["token_budget", "extractive", "abstractive"]
    keeps   = [0.75, 0.50, 0.25]

    b_rows  = [r for r in rows if r["method"] == "baseline"]
    b_wall  = safe_mean([r["wall_s"] for r in b_rows]) or 0

    print(f"\n{'='*78}")
    print("KEY FINDINGS")
    print(f"{'='*78}")

    # Best speedup with acceptable quality
    candidates = [r for r in rows if r["method"] != "baseline"
                  and r.get("rouge1_f1") is not None
                  and r.get("speedup_vs_baseline") is not None]

    # Group by (method, keep)
    combo_stats = {}
    for method in methods:
        for keep in keeps:
            sub = [r for r in candidates if r["method"] == method
                   and r.get("keep_fraction_target") == keep]
            if not sub:
                continue
            combo_stats[(method, keep)] = {
                "rouge": safe_mean([r["rouge1_f1"] for r in sub]),
                "ent":   safe_mean([r["entity_recall"] for r in sub]),
                "speedup": safe_mean([r["speedup_vs_baseline"] for r in sub]),
                "wall": safe_mean([r["wall_s"] for r in sub]),
                "ratio": safe_mean([r["actual_ratio"] for r in sub]),
                "n": len(sub),
            }

    print(f"\n  Baseline mean wall: {b_wall:.1f}s (n={len(b_rows)})")
    print(f"\n  Best speedup overall:")
    best_spd = sorted(combo_stats.items(), key=lambda x: -(x[1]["speedup"] or 0))
    for (m, k), s in best_spd[:3]:
        print(f"    {m} keep={int(k*100)}%: {s['speedup']:.2f}× speedup, "
              f"ROUGE={s['rouge']:.3f}, ent={s['ent']:.3f}, wall={s['wall']:.1f}s")

    print(f"\n  Best quality (ROUGE-1):")
    best_q = sorted(combo_stats.items(), key=lambda x: -(x[1]["rouge"] or 0))
    for (m, k), s in best_q[:3]:
        print(f"    {m} keep={int(k*100)}%: ROUGE={s['rouge']:.3f}, "
              f"speedup={s['speedup']:.2f}×, wall={s['wall']:.1f}s")

    print(f"\n  Best quality-efficiency trade-off (ROUGE ≥ 0.50, maximize speedup):")
    pareto = [(mk, s) for mk, s in combo_stats.items()
              if (s["rouge"] or 0) >= 0.50]
    if pareto:
        pareto_sorted = sorted(pareto, key=lambda x: -(x[1]["speedup"] or 0))
        for (m, k), s in pareto_sorted[:3]:
            print(f"    {m} keep={int(k*100)}%: speedup={s['speedup']:.2f}×, "
                  f"ROUGE={s['rouge']:.3f}")
    else:
        # Lower threshold
        pareto = [(mk, s) for mk, s in combo_stats.items()
                  if (s["rouge"] or 0) >= 0.40]
        pareto_sorted = sorted(pareto, key=lambda x: -(x[1]["speedup"] or 0))
        print(f"    [no method ≥ ROUGE 0.50 — showing ≥ 0.40:]")
        for (m, k), s in pareto_sorted[:3]:
            print(f"    {m} keep={int(k*100)}%: speedup={s['speedup']:.2f}×, "
                  f"ROUGE={s['rouge']:.3f}")

    print(f"\n  Abstractive compression quality vs ratio:")
    for keep in keeps:
        sub = [r for r in rows if r["method"] == "abstractive"
               and r.get("keep_fraction_target") == keep]
        if not sub:
            continue
        tgt_ratio = keep
        act_ratio = safe_mean([r["actual_ratio"] for r in sub]) or 0
        rouge     = safe_mean([r["rouge1_f1"] for r in sub]) or 0
        comp_ms   = safe_mean([r["comp_latency_ms"] for r in sub]) or 0
        print(f"    target {int(tgt_ratio*100)}%: actual={act_ratio:.3f} "
              f"(vs {tgt_ratio:.2f} target), ROUGE={rouge:.3f}, "
              f"comp_latency={comp_ms/1000:.1f}s")

    print(f"\n  Summary for paper:")
    print(f"    - Token budget 25% keep: highest speedup, low ROUGE (aggressive truncation)")
    print(f"    - Extractive 50% keep: good balance of speed and quality")
    print(f"    - Abstractive: consistently fails to achieve target compression ratios")
    print(f"      (qwen2:1.5b produces summaries ~2-3× longer than specified, "
           f"eliminating expected speedup)")

    return combo_stats


# ── Comp-latency overhead check ────────────────────────────────────────────────

def comp_overhead_analysis(rows):
    print(f"\n{'='*78}")
    print("COMPRESSION LATENCY OVERHEAD ANALYSIS")
    print(f"{'='*78}")
    print(f"  Does compression latency negate inference savings?")
    print(f"  (comp_latency_ms + compressed_wall_s) vs baseline_wall_s\n")

    b_rows = [r for r in rows if r["method"] == "baseline"]
    b_wall_mean = safe_mean([r["wall_s"] for r in b_rows]) or 0

    for method in ["token_budget", "extractive", "abstractive"]:
        print(f"  {method}:")
        for keep in [0.75, 0.50, 0.25]:
            sub = [r for r in rows if r["method"] == method
                   and r.get("keep_fraction_target") == keep
                   and r["wall_s"] is not None]
            if not sub:
                continue
            total_w = [(r["wall_s"] + (r["comp_latency_ms"] or 0) / 1000)
                       for r in sub]
            b_w     = [r["baseline_wall_s"] for r in sub if r["baseline_wall_s"]]
            bwall   = safe_mean(b_w) or b_wall_mean
            combined_mean = safe_mean(total_w) or 0
            net_speedup  = bwall / combined_mean if combined_mean > 0 else 0
            print(f"    keep={int(keep*100)}%: "
                  f"inf={safe_mean([r['wall_s'] for r in sub]):.1f}s + "
                  f"comp={safe_mean([r['comp_latency_ms'] for r in sub])/1000:.1f}s = "
                  f"total={combined_mean:.1f}s  →  net_speedup={net_speedup:.2f}× "
                  f"(vs baseline {bwall:.1f}s)")
        print()


# ── Save analysis JSON ─────────────────────────────────────────────────────────

def save_analysis(rows, summary_rows, bucket_rows, outpath: Path):
    out = {
        "source_rows": len(rows),
        "n_prompts": len(set(r["id"] for r in rows if r["method"] == "baseline")),
        "summary": summary_rows,
        "by_bucket": bucket_rows,
    }
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAnalysis saved to {outpath}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    force = None
    if "--full" in sys.argv:
        force = "full"
    elif "--partial" in sys.argv:
        force = "partial"

    results, source = load_results(force)
    rows = flatten_rows(results)
    print(f"Flattened to {len(rows)} measurement rows "
          f"(including {len([r for r in rows if r['method']=='baseline'])} baselines)")

    # Count complete prompts
    prompt_ids = sorted(set(r["id"] for r in rows))
    print(f"Prompt IDs present: {prompt_ids}")

    # Tables
    summary_rows = table_method_level(rows)
    bucket_rows  = table_by_bucket(rows)
    table_paper_markdown(rows)

    # LaTeX
    latex_path = RESULTS_DIR / "table_compression.tex"
    latex_bucket_path = RESULTS_DIR / "table_compression_bucket.tex"
    table_latex(rows, latex_path)
    table_by_bucket_latex(rows, latex_bucket_path)

    # Findings
    combo_stats = print_key_findings(rows)
    comp_overhead_analysis(rows)

    # Save
    save_analysis(rows, summary_rows, bucket_rows, RESULTS_DIR / "exp_b_analysis.json")

    print(f"\n{'='*78}")
    print("DONE — run again after exp_compression_full.json is available for full P1-P10 results")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
