#!/usr/bin/env python3
"""
Benchmark Summary — Aggregate Results
======================================
Generates a comprehensive summary of the benchmark results for inclusion
in the review manuscript.

Inputs:
    ../RESULTS/SCORES/model_scores.csv
    ../RESULTS/SCORES/probe_scores.csv
    ../RESULTS/SCORES/disease_scores.csv
    ../RESULTS/sii_correlation_results.json
    ../DATA/heim_diseases.csv
    ../DATA/ground-truth/disease_ground_truth.csv

Outputs:
    ../RESULTS/benchmark_summary.json
    ../RESULTS/benchmark_summary.txt       — human-readable report
    ../RESULTS/manuscript_statistics.txt   — copy-paste stats for manuscript

Author: Manuel Corpas
Date: February 2026
"""

import csv
import json
import statistics
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "RESULTS"
SCORES_DIR = RESULTS_DIR / "SCORES"
DATA_DIR = Path(__file__).parent.parent / "DATA"

MODEL_SCORES = SCORES_DIR / "model_scores.csv"
PROBE_SCORES = SCORES_DIR / "probe_scores.csv"
DISEASE_SCORES = SCORES_DIR / "disease_scores.csv"
CORRELATION_RESULTS = RESULTS_DIR / "sii_correlation_results.json"
HEIM_CSV = DATA_DIR / "heim_diseases.csv"
GROUND_TRUTH = DATA_DIR / "ground-truth" / "disease_ground_truth.csv"

OUTPUT_JSON = RESULTS_DIR / "benchmark_summary.json"
OUTPUT_TXT = RESULTS_DIR / "benchmark_summary.txt"
MANUSCRIPT_TXT = RESULTS_DIR / "manuscript_statistics.txt"


def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def main():
    print("=" * 70)
    print("Benchmark Summary — Generating Report")
    print("=" * 70)

    summary = {}
    report_lines = []

    def section(title):
        report_lines.append(f"\n{'=' * 70}")
        report_lines.append(title)
        report_lines.append("=" * 70)

    # -----------------------------------------------------------------------
    # Overview
    # -----------------------------------------------------------------------
    section("OVERVIEW")

    heim = load_csv(HEIM_CSV)
    n_diseases = len(heim)
    n_models = 6
    n_probes = 10
    n_queries = n_diseases * n_models * n_probes

    summary["n_diseases"] = n_diseases
    summary["n_models"] = n_models
    summary["n_probes"] = n_probes
    summary["n_queries"] = n_queries

    report_lines.append(f"Diseases evaluated:  {n_diseases}")
    report_lines.append(f"Models tested:       {n_models}")
    report_lines.append(f"Probes per disease:  {n_probes}")
    report_lines.append(f"Total queries:       {n_queries}")
    report_lines.append(f"Evaluation metric:   Top-50 semantic similarity (PubMedBERT)")

    # -----------------------------------------------------------------------
    # Model performance
    # -----------------------------------------------------------------------
    section("MODEL PERFORMANCE")

    try:
        models = load_csv(MODEL_SCORES)
        models_sorted = sorted(models,
                               key=lambda m: float(m["mean_semantic_similarity"]),
                               reverse=True)

        report_lines.append(
            f"\n{'Model':<25s} {'Similarity':>12s} {'SD':>8s} {'N':>8s}")
        report_lines.append("-" * 55)
        for m in models_sorted:
            report_lines.append(
                f"{m['display_name']:<25s} {m['mean_semantic_similarity']:>12s} "
                f"{m['std_semantic_similarity']:>8s} {m['n_scores']:>8s}"
            )
        summary["model_rankings"] = [
            {"model": m["display_name"],
             "mean_semantic_similarity": m["mean_semantic_similarity"],
             "n_scores": m["n_scores"]}
            for m in models_sorted
        ]
    except FileNotFoundError:
        report_lines.append("  [Model scores not yet available]")

    # -----------------------------------------------------------------------
    # Probe-level performance
    # -----------------------------------------------------------------------
    section("PROBE-LEVEL PERFORMANCE (Knowledge Dimensions)")

    try:
        probes = load_csv(PROBE_SCORES)
        probes_sorted = sorted(probes,
                               key=lambda p: float(p["mean_semantic_similarity"]),
                               reverse=True)

        report_lines.append(
            f"\n{'Probe':<30s} {'Dimension':<22s} {'Similarity':>12s} {'SD':>8s}")
        report_lines.append("-" * 75)
        for p in probes_sorted:
            report_lines.append(
                f"{p['probe']:<30s} {p['dimension']:<22s} "
                f"{p['mean_semantic_similarity']:>12s} "
                f"{p['std_semantic_similarity']:>8s}"
            )
        summary["probe_rankings"] = [
            {"probe": p["probe"], "dimension": p["dimension"],
             "mean_semantic_similarity": p["mean_semantic_similarity"]}
            for p in probes_sorted
        ]
    except FileNotFoundError:
        report_lines.append("  [Probe scores not yet available]")

    # -----------------------------------------------------------------------
    # SII correlation
    # -----------------------------------------------------------------------
    section("SII — SEMANTIC SIMILARITY CORRELATION")

    try:
        with open(CORRELATION_RESULTS, "r") as f:
            corr = json.load(f)

        report_lines.append(f"\nH1: SII correlates with LLM-literature alignment")
        report_lines.append(
            f"  Pearson r:  {corr['h1_pearson_r']:.4f}  "
            f"(p={corr['h1_pearson_p']:.2e})")
        report_lines.append(
            f"  Spearman ρ: {corr['h1_spearman_rho']:.4f}  "
            f"(p={corr['h1_spearman_p']:.2e})")

        report_lines.append(f"\nH2: SII predicts beyond publication volume")
        report_lines.append(
            f"  Partial r (SII | papers): {corr['h2_partial_r']:.4f}  "
            f"(p={corr['h2_partial_p']:.2e})")
        report_lines.append(
            f"  log(papers) alone:        {corr['h2_papers_r']:.4f}  "
            f"(p={corr['h2_papers_p']:.2e})")

        if "h2_regression" in corr:
            reg = corr["h2_regression"]
            report_lines.append(
                f"\n  Multiple regression (SII + KTP + RCC + log_papers):")
            report_lines.append(
                f"    R² = {reg['r_squared']:.4f}, adj R² = {reg['adj_r_squared']:.4f}")
            report_lines.append(
                f"    F = {reg['f_statistic']:.2f}, p = {reg['f_pvalue']:.2e}")

        if "ntd_effect" in corr:
            eff = corr["ntd_effect"]
            report_lines.append(f"\n  NTD vs non-NTD effect size:")
            report_lines.append(f"    NTD mean:     {eff['ntd_mean']:.6f}")
            report_lines.append(f"    Non-NTD mean: {eff['non_ntd_mean']:.6f}")
            report_lines.append(f"    Cohen's d:    {eff['cohens_d']:.4f}")
            report_lines.append(f"    t-test p:     {eff['t_pval']:.2e}")

        if "probe_correlations" in corr:
            report_lines.append(
                f"\n  Probe-level SII sensitivity (most → least sensitive):")
            ranked = sorted(corr["probe_correlations"].items(),
                            key=lambda x: x[1]["r"])
            for probe, vals in ranked:
                sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 \
                    else "*" if vals["p"] < 0.05 else "ns"
                report_lines.append(
                    f"    {probe:<25s} r={vals['r']:+.4f}  p={vals['p']:.2e}  {sig}")

        summary["correlation"] = corr
    except FileNotFoundError:
        report_lines.append("  [Correlation results not yet available]")

    # -----------------------------------------------------------------------
    # Disease-level highlights
    # -----------------------------------------------------------------------
    section("DISEASE-LEVEL HIGHLIGHTS")

    try:
        diseases = load_csv(DISEASE_SCORES)
        heim_dict = {h["disease"]: h for h in heim}

        by_sim = sorted(diseases,
                        key=lambda d: float(d["mean_semantic_similarity"]))

        report_lines.append(
            "\nBottom 10 diseases (lowest LLM-literature alignment):")
        for d in by_sim[:10]:
            sii_val = heim_dict.get(d["disease"], {}).get("sii", "?")
            n_abs = d.get("n_abstracts", "?")
            report_lines.append(
                f"  {d['disease']:<45s} sim={d['mean_semantic_similarity']}  "
                f"SII={sii_val}  abstracts={n_abs}")

        report_lines.append(
            "\nTop 10 diseases (highest LLM-literature alignment):")
        for d in by_sim[-10:]:
            sii_val = heim_dict.get(d["disease"], {}).get("sii", "?")
            n_abs = d.get("n_abstracts", "?")
            report_lines.append(
                f"  {d['disease']:<45s} sim={d['mean_semantic_similarity']}  "
                f"SII={sii_val}  abstracts={n_abs}")

    except FileNotFoundError:
        report_lines.append("  [Disease scores not yet available]")

    # -----------------------------------------------------------------------
    # Manuscript-ready statistics
    # -----------------------------------------------------------------------
    manuscript_lines = [
        "MANUSCRIPT-READY STATISTICS",
        "=" * 40,
        "(Copy-paste into manuscript draft)",
        "",
        f"We evaluated {n_models} frontier LLMs across {n_diseases} GBD disease "
        f"categories and {n_probes} knowledge dimensions, totalling {n_queries:,} queries.",
        "",
        "Each LLM response was embedded using PubMedBERT and compared to the",
        "published PubMed literature for its disease by computing the mean cosine",
        "similarity to the 50 most relevant abstracts.",
        "",
    ]

    try:
        corr = summary.get("correlation", {})
        if corr:
            manuscript_lines.extend([
                "SII-performance correlation:",
                f"  Pearson r = {corr.get('h1_pearson_r', 0):.3f}, "
                f"p = {corr.get('h1_pearson_p', 1):.1e}",
                f"  Spearman ρ = {corr.get('h1_spearman_rho', 0):.3f}, "
                f"p = {corr.get('h1_spearman_p', 1):.1e}",
                "",
                "Partial correlation (controlling for publication volume):",
                f"  r = {corr.get('h2_partial_r', 0):.3f}, "
                f"p = {corr.get('h2_partial_p', 1):.1e}",
                "",
                f"Publication volume alone: r = {corr.get('h2_papers_r', 0):.3f}, "
                f"p = {corr.get('h2_papers_p', 1):.1e}",
                "",
            ])
            if "h2_regression" in corr:
                reg = corr["h2_regression"]
                manuscript_lines.extend([
                    "Multiple regression (SII + KTP + RCC + log papers):",
                    f"  R² = {reg['r_squared']:.3f}, "
                    f"F({reg['k']},{reg['n']-reg['k']-1}) = {reg['f_statistic']:.1f}, "
                    f"p = {reg['f_pvalue']:.1e}",
                    "",
                ])
            if "probe_correlations" in corr:
                manuscript_lines.append("Probe-level SII-similarity correlations:")
                ranked = sorted(corr["probe_correlations"].items(),
                                key=lambda x: x[1]["r"])
                for probe, vals in ranked:
                    manuscript_lines.append(
                        f"  {probe}: r = {vals['r']:.3f}, p = {vals['p']:.1e}")
                manuscript_lines.append("")
    except (TypeError, ValueError):
        manuscript_lines.append("  [Statistics will be available after evaluation]")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    report_text = "\n".join(report_lines)
    manuscript_text = "\n".join(manuscript_lines)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_JSON}")

    with open(OUTPUT_TXT, "w") as f:
        f.write(report_text)
    print(f"  Saved: {OUTPUT_TXT}")

    with open(MANUSCRIPT_TXT, "w") as f:
        f.write(manuscript_text)
    print(f"  Saved: {MANUSCRIPT_TXT}")

    print(report_text)


if __name__ == "__main__":
    main()
