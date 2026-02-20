#!/usr/bin/env python3
"""
Correlate SII with LLM Performance (Semantic Similarity)
=========================================================
The central analysis: does the Semantic Isolation Index (SII) from the HEIM
framework predict how well LLM responses align with published literature?

Measure: cosine similarity between PubMedBERT embeddings of LLM responses
and the centroid of PubMed abstracts for each disease.

Hypotheses:
  H1: SII correlates negatively with semantic similarity
      (diseases isolated in literature → LLM responses less aligned)
  H2: SII predicts LLM performance beyond what publication count alone explains

Statistical tests:
  - Pearson r and Spearman rho (SII vs mean similarity), n=175
  - Partial correlation controlling for publication volume
  - Multiple regression: SII + KTP + RCC + log(n_papers) -> similarity
  - Probe-level analysis: which knowledge dimensions show steepest SII gradient
  - Effect size: Cohen's d for NTD vs non-NTD diseases

Inputs:
    ../DATA/heim_diseases.csv
    ../RESULTS/SCORES/disease_scores.csv
    ../RESULTS/SCORES/all_scores.json

Outputs:
    ../RESULTS/sii_correlation_results.json
    ../RESULTS/sii_correlation_results.csv

Author: Manuel Corpas
Date: February 2026
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "DATA"
SCORES_DIR = Path(__file__).parent.parent / "RESULTS" / "SCORES"
RESULTS_DIR = Path(__file__).parent.parent / "RESULTS"

HEIM_CSV = DATA_DIR / "heim_diseases.csv"
DISEASE_SCORES_CSV = SCORES_DIR / "disease_scores.csv"
ALL_SCORES_JSON = SCORES_DIR / "all_scores.json"

OUTPUT_JSON = RESULTS_DIR / "sii_correlation_results.json"
OUTPUT_CSV = RESULTS_DIR / "sii_correlation_results.csv"


def load_heim():
    """Load HEIM metrics for all 175 diseases."""
    data = {}
    with open(HEIM_CSV, "r") as f:
        for row in csv.DictReader(f):
            data[row["disease"]] = {
                "sii": float(row["sii"]),
                "ktp": float(row["ktp"]),
                "rcc": float(row["rcc"]),
                "n_papers": int(row["n_papers"]),
                "log_papers": math.log10(int(row["n_papers"])),
            }
    return data


def load_disease_scores():
    """Load mean semantic similarity per disease."""
    data = {}
    with open(DISEASE_SCORES_CSV, "r") as f:
        for row in csv.DictReader(f):
            data[row["disease"]] = {
                "mean_semantic_similarity": float(row["mean_semantic_similarity"]),
            }
    return data


def load_probe_scores():
    """Load individual scores for probe-level analysis."""
    with open(ALL_SCORES_JSON, "r") as f:
        return json.load(f)


def partial_correlation(x, y, z):
    """Partial correlation between x and y, controlling for z."""
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p


def multiple_regression(X, y):
    """OLS multiple regression. Returns coefficients, R², F-stat, p-value."""
    n, k = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    y_hat = X_aug @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else 0

    ms_reg = (ss_tot - ss_res) / k if k > 0 else 0
    ms_res = ss_res / (n - k - 1) if n > k + 1 else 1
    f_stat = ms_reg / ms_res if ms_res > 0 else 0
    f_pvalue = 1 - stats.f.cdf(f_stat, k, n - k - 1) if n > k + 1 else 1

    return {
        "coefficients": beta.tolist(),
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_statistic": f_stat,
        "f_pvalue": f_pvalue,
        "n": n,
        "k": k,
    }


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    print("=" * 70)
    print("SII — Semantic Similarity Correlation Analysis")
    print("=" * 70)

    # Load data
    heim = load_heim()
    scores = load_disease_scores()

    # Merge on disease name
    diseases = sorted(set(heim.keys()) & set(scores.keys()))
    print(f"\n  Matched diseases: {len(diseases)} / {len(heim)} HEIM diseases")

    if len(diseases) < 10:
        print("ERROR: Too few matched diseases.")
        raise SystemExit(1)

    # Build arrays
    sii = np.array([heim[d]["sii"] for d in diseases])
    ktp = np.array([heim[d]["ktp"] for d in diseases])
    rcc = np.array([heim[d]["rcc"] for d in diseases])
    log_papers = np.array([heim[d]["log_papers"] for d in diseases])
    similarity = np.array([scores[d]["mean_semantic_similarity"] for d in diseases])

    results = {"n_diseases": len(diseases)}

    # -----------------------------------------------------------------------
    # H1: SII correlates with semantic similarity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("H1: SII correlates negatively with semantic similarity")
    print("=" * 70)

    r_p, p_p = stats.pearsonr(sii, similarity)
    print(f"\n  Pearson r  (SII vs similarity): {r_p:.4f}  p={p_p:.2e}")

    r_s, p_s = stats.spearmanr(sii, similarity)
    print(f"  Spearman ρ (SII vs similarity): {r_s:.4f}  p={p_s:.2e}")

    results["h1_pearson_r"] = r_p
    results["h1_pearson_p"] = p_p
    results["h1_spearman_rho"] = r_s
    results["h1_spearman_p"] = p_s

    # -----------------------------------------------------------------------
    # H2: SII predicts beyond publication volume
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("H2: SII predicts beyond publication count")
    print("=" * 70)

    r_partial, p_partial = partial_correlation(sii, similarity, log_papers)
    print(f"\n  Partial r (SII vs similarity | log_papers): {r_partial:.4f}  p={p_partial:.2e}")

    r_papers, p_papers = stats.pearsonr(log_papers, similarity)
    print(f"  Pearson r (log_papers vs similarity):        {r_papers:.4f}  p={p_papers:.2e}")

    results["h2_partial_r"] = r_partial
    results["h2_partial_p"] = p_partial
    results["h2_papers_r"] = r_papers
    results["h2_papers_p"] = p_papers

    # Multiple regression
    X = np.column_stack([sii, ktp, rcc, log_papers])
    reg = multiple_regression(X, similarity)
    if reg:
        print(f"\n  Multiple regression (SII + KTP + RCC + log_papers):")
        print(f"    R²:           {reg['r_squared']:.4f}")
        print(f"    Adjusted R²:  {reg['adj_r_squared']:.4f}")
        print(f"    F-statistic:  {reg['f_statistic']:.2f}  p={reg['f_pvalue']:.2e}")
        coef_names = ["intercept", "SII", "KTP", "RCC", "log_papers"]
        for name, coef in zip(coef_names, reg["coefficients"]):
            print(f"    β_{name}: {coef:.6f}")
        results["h2_regression"] = reg

    # -----------------------------------------------------------------------
    # Effect size: NTD vs non-NTD
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Effect Size: NTD vs non-NTD diseases")
    print("=" * 70)

    # Load categories
    gt_csv = Path(__file__).parent.parent / "DATA" / "ground-truth" / "disease_ground_truth.csv"
    categories = {}
    if gt_csv.exists():
        with open(gt_csv, "r") as f:
            for row in csv.DictReader(f):
                categories[row["disease"]] = row.get("category", "Other")

    ntd_sim = [scores[d]["mean_semantic_similarity"] for d in diseases
               if categories.get(d) == "NTD"]
    non_ntd_sim = [scores[d]["mean_semantic_similarity"] for d in diseases
                   if categories.get(d) != "NTD"]

    if ntd_sim and non_ntd_sim:
        d_val = cohens_d(np.array(ntd_sim), np.array(non_ntd_sim))
        t_stat, t_pval = stats.ttest_ind(ntd_sim, non_ntd_sim)
        u_stat, u_pval = stats.mannwhitneyu(ntd_sim, non_ntd_sim, alternative="two-sided")
        print(f"\n  NTD mean similarity:     {np.mean(ntd_sim):.6f} (n={len(ntd_sim)})")
        print(f"  Non-NTD mean similarity: {np.mean(non_ntd_sim):.6f} (n={len(non_ntd_sim)})")
        print(f"  Cohen's d: {d_val:.4f}")
        print(f"  t-test: t={t_stat:.3f}, p={t_pval:.2e}")
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_pval:.2e}")
        results["ntd_effect"] = {
            "ntd_mean": float(np.mean(ntd_sim)),
            "non_ntd_mean": float(np.mean(non_ntd_sim)),
            "cohens_d": d_val,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "mw_u": float(u_stat),
            "mw_pval": float(u_pval),
        }

    # -----------------------------------------------------------------------
    # Probe-level analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Probe-Level Analysis: SII gradient by knowledge dimension")
    print("=" * 70)

    try:
        all_scores = load_probe_scores()
        valid = [s for s in all_scores if s.get("semantic_similarity") is not None]

        probe_disease_scores = {}
        for s in valid:
            key = (s["probe"], s["disease"])
            if key not in probe_disease_scores:
                probe_disease_scores[key] = []
            probe_disease_scores[key].append(s["semantic_similarity"])

        probe_results = {}
        probes_sorted = sorted(set(s["probe"] for s in valid))

        for probe in probes_sorted:
            probe_sii = []
            probe_score = []
            for d in diseases:
                key = (probe, d)
                if key in probe_disease_scores:
                    probe_sii.append(heim[d]["sii"])
                    probe_score.append(np.mean(probe_disease_scores[key]))

            if len(probe_sii) >= 10:
                r, p = stats.pearsonr(probe_sii, probe_score)
                dim = next(
                    (s["dimension"] for s in valid if s["probe"] == probe), "Unknown"
                )
                probe_results[probe] = {"dimension": dim, "r": r, "p": p, "n": len(probe_sii)}
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {probe:25s} ({dim:20s}): r={r:+.4f}  p={p:.2e}  {sig}")

        results["probe_correlations"] = probe_results

        # Knowledge chain analysis
        print("\n  Knowledge chain (ordered by SII sensitivity):")
        ranked = sorted(probe_results.items(), key=lambda x: x[1]["r"])
        for i, (probe, vals) in enumerate(ranked, 1):
            print(f"    {i}. {probe} (r={vals['r']:+.4f})")

    except FileNotFoundError:
        print("  Skipping probe-level analysis (all_scores.json not found)")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_JSON}")

    # Save disease-level merged data for plotting
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["disease", "sii", "ktp", "rcc", "log_papers", "n_papers",
                     "mean_semantic_similarity"])
        for i, d in enumerate(diseases):
            w.writerow([
                d, sii[i], ktp[i], rcc[i], log_papers[i], heim[d]["n_papers"],
                similarity[i],
            ])
    print(f"  Saved: {OUTPUT_CSV}")

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
