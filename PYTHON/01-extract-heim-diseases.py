#!/usr/bin/env python3
"""
Extract HEIM Diseases and Metrics
==================================
Loads all 175 GBD diseases with their Semantic Isolation Index (SII),
Knowledge Trajectory Profile (KTP), Research Completeness Category (RCC),
and publication counts from the HEIM disease_metrics.csv.

Produces a clean, sorted reference file for downstream benchmark scripts.

Outputs:
    ../DATA/heim_diseases.csv    — sorted by SII descending (most isolated first)
    ../DATA/heim_diseases.json

Author: Manuel Corpas
Date: February 2026
"""

import csv
import json
from pathlib import Path

HEIM_METRICS = Path(
    "/Users/superintelligent/Library/Mobile Documents/"
    "com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS/"
    "ANALYSIS/05-03-SEMANTIC-METRICS/disease_metrics.csv"
)
OUTPUT_DIR = Path(__file__).parent.parent / "DATA"
OUTPUT_CSV = OUTPUT_DIR / "heim_diseases.csv"
OUTPUT_JSON = OUTPUT_DIR / "heim_diseases.json"


def main():
    print("Loading HEIM disease metrics...")
    diseases = []

    with open(HEIM_METRICS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            diseases.append({
                "disease": row["disease"],
                "disease_display": row["disease"].replace("_", " "),
                "n_papers": int(row["n_papers"]),
                "sii": float(row["sii"]),
                "ktp": float(row["ktp"]),
                "rcc": float(row["rcc"]),
                "mean_drift": float(row["mean_drift"]),
            })

    print(f"  Loaded {len(diseases)} diseases")

    # Sort by SII descending (most isolated first)
    diseases.sort(key=lambda d: d["sii"], reverse=True)

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["disease", "disease_display", "n_papers", "sii", "ktp", "rcc", "mean_drift"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diseases)
    print(f"  Saved: {OUTPUT_CSV}")

    # Write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(diseases, f, indent=2)
    print(f"  Saved: {OUTPUT_JSON}")

    # Print top 10 most isolated and bottom 10 least isolated
    print("\n  Top 10 most semantically isolated diseases:")
    for d in diseases[:10]:
        print(f"    SII={d['sii']:.6f}  n={d['n_papers']:>7d}  {d['disease_display']}")

    print("\n  Top 10 least semantically isolated diseases:")
    for d in diseases[-10:]:
        print(f"    SII={d['sii']:.6f}  n={d['n_papers']:>7d}  {d['disease_display']}")

    # Correlation between SII and paper count (quick check)
    import math
    log_papers = [math.log10(d["n_papers"]) for d in diseases]
    sii_vals = [d["sii"] for d in diseases]
    n = len(diseases)
    mean_x = sum(log_papers) / n
    mean_y = sum(sii_vals) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_papers, sii_vals)) / n
    std_x = (sum((x - mean_x) ** 2 for x in log_papers) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in sii_vals) / n) ** 0.5
    r = cov / (std_x * std_y) if std_x * std_y > 0 else 0
    print(f"\n  Pearson r (log10(n_papers) vs SII): {r:.4f}")
    print("  (Negative r expected: fewer papers → higher isolation)")


if __name__ == "__main__":
    main()
