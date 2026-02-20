#!/usr/bin/env python3
"""
Evaluate LLM Responses — Top-k Semantic Similarity to Literature
=================================================================
For each LLM response, compute its average cosine similarity to the k most
similar PubMed abstracts for that disease (using PubMedBERT embeddings).

This avoids the centroid-dilution problem: for well-studied diseases with
diverse literatures, the centroid is blurred and any single response scores
low.  By comparing to the top-k *most relevant* abstracts instead, we measure
how well the LLM captures actual published knowledge regardless of how broad
or narrow that literature is.

Method:
  1. Embed each LLM response with PubMedBERT
  2. For each response, load the pre-computed abstract embeddings for its disease
  3. Compute cosine similarity to ALL abstracts
  4. Score = mean similarity to the top-k most similar abstracts

Inputs:
    HEIM embeddings: 07-EHR-LINKED-BIOBANKS/DATA/05-SEMANTIC/EMBEDDINGS/
    LLM responses:   ../RESULTS/RESPONSES/*.json

Outputs:
    ../RESULTS/SCORES/all_scores.json
    ../RESULTS/SCORES/disease_scores.csv
    ../RESULTS/SCORES/model_scores.csv
    ../RESULTS/SCORES/probe_scores.csv

Author: Manuel Corpas
Date: February 2026
"""

import csv
import json
import sys
import statistics as stats_mod
from pathlib import Path

import numpy as np
import h5py
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_K = 50  # Number of most-similar abstracts to average over

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
RESPONSES_DIR = BASE / "RESULTS" / "RESPONSES"
SCORES_DIR = BASE / "RESULTS" / "SCORES"
GT_CSV = BASE / "DATA" / "ground-truth" / "disease_ground_truth.csv"

HEIM_EMBEDDINGS = Path(
    "/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/"
    "PUBLICATIONS/07-EHR-LINKED-BIOBANKS/DATA/05-SEMANTIC/EMBEDDINGS"
)

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


# ---------------------------------------------------------------------------
# PubMedBERT embedding
# ---------------------------------------------------------------------------

def load_pubmedbert(device):
    """Load the PubMedBERT tokenizer and model."""
    print(f"\n  Loading PubMedBERT on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print("  Model loaded")
    return tokenizer, model


def embed_texts(texts, tokenizer, model, device, batch_size=32):
    """
    Embed a list of texts with PubMedBERT.
    Returns np.array of shape (len(texts), 768), L2-normalised.
    """
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, len(texts), batch_size)):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)
        if (i + 1) % 50 == 0:
            print(f"    Embedded batch {i + 1}/{n_batches}")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    # L2 normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    return embeddings


# ---------------------------------------------------------------------------
# Top-k similarity scoring
# ---------------------------------------------------------------------------

def load_disease_embeddings(disease):
    """
    Load all abstract embeddings for a disease from HDF5.
    Returns L2-normalised np.array of shape (n_papers, 768).
    """
    h5_path = HEIM_EMBEDDINGS / disease / "embeddings.h5"
    if not h5_path.exists():
        return None

    with h5py.File(h5_path, "r") as f:
        n = f["embeddings"].shape[0]
        # Load in chunks for very large files
        if n > 100000:
            chunk_size = 50000
            chunks = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunks.append(f["embeddings"][start:end][:])
            emb = np.vstack(chunks)
        else:
            emb = f["embeddings"][:]

    # L2 normalise
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    return emb


def score_top_k(response_embeddings, abstract_embeddings, k=TOP_K):
    """
    For each response embedding, compute mean cosine similarity to the
    k most similar abstract embeddings.

    response_embeddings: (n_responses, 768)
    abstract_embeddings: (n_abstracts, 768)

    Returns: np.array of shape (n_responses,)
    """
    n_abstracts = abstract_embeddings.shape[0]
    actual_k = min(k, n_abstracts)

    # Cosine similarity matrix (already L2-normalised, so just dot product)
    # (n_responses, 768) @ (768, n_abstracts) -> (n_responses, n_abstracts)
    sim_matrix = response_embeddings @ abstract_embeddings.T

    # For each response, get mean of top-k similarities
    if actual_k == n_abstracts:
        # Use all abstracts
        return sim_matrix.mean(axis=1)

    # Partial sort to find top-k per row (more efficient than full sort)
    # np.partition puts the k largest values in the last k positions
    partitioned = np.partition(sim_matrix, -actual_k, axis=1)
    top_k_sims = partitioned[:, -actual_k:]
    return top_k_sims.mean(axis=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_responses():
    """Load all valid response JSON files."""
    responses = []
    for fpath in sorted(RESPONSES_DIR.glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("is_error", False):
            data["response_file"] = fpath.name
            responses.append(data)
    return responses


def load_categories():
    """Load disease categories from ground truth."""
    cats = {}
    if GT_CSV.exists():
        with open(GT_CSV, "r") as f:
            for row in csv.DictReader(f):
                cats[row["disease"]] = row.get("category", "Other")
    return cats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("African Health LLM Benchmark — Top-k Semantic Similarity")
    print(f"(Mean cosine similarity to {TOP_K} most relevant PubMed abstracts)")
    print("=" * 70)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load PubMedBERT
    tokenizer, model = load_pubmedbert(device)

    # Load responses
    print("\n  Loading responses...")
    responses = load_responses()
    print(f"  {len(responses)} valid responses")

    if not responses:
        print("ERROR: No responses found.")
        raise SystemExit(1)

    categories = load_categories()

    # Group responses by disease
    disease_groups = {}
    for i, resp in enumerate(responses):
        d = resp["disease"]
        if d not in disease_groups:
            disease_groups[d] = []
        disease_groups[d].append((i, resp))
    print(f"  {len(disease_groups)} diseases represented")

    # Embed ALL response texts first
    print(f"\n  Embedding {len(responses)} LLM responses with PubMedBERT...")
    response_texts = [r["response"] for r in responses]
    response_embeddings = embed_texts(
        response_texts, tokenizer, model, device, batch_size=32
    )
    print("  Embedding complete")

    # Free model from memory
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()

    # Score disease by disease
    print(f"\n  Scoring responses against PubMed abstracts (top-{TOP_K})...")
    all_scores = [None] * len(responses)  # Pre-allocate to maintain order
    diseases_processed = 0

    for disease, group in sorted(disease_groups.items()):
        # Load abstract embeddings for this disease
        abstract_emb = load_disease_embeddings(disease)
        if abstract_emb is None:
            # No embeddings for this disease — skip
            for idx, resp in group:
                all_scores[idx] = {
                    "model": resp["model"],
                    "model_display": resp.get("model_display", resp["model"]),
                    "probe": resp["probe"],
                    "dimension": resp["dimension"],
                    "disease": resp["disease"],
                    "disease_display": resp["disease_display"],
                    "category": categories.get(resp["disease"], "Other"),
                    "semantic_similarity": None,
                    "n_abstracts": 0,
                    "total_words": len(resp["response"].split()),
                }
            continue

        # Get response embeddings for this disease
        indices = [idx for idx, _ in group]
        disease_resp_emb = response_embeddings[indices]

        # Compute top-k similarities
        top_k_scores = score_top_k(disease_resp_emb, abstract_emb, k=TOP_K)

        # Store results
        for j, (idx, resp) in enumerate(group):
            all_scores[idx] = {
                "model": resp["model"],
                "model_display": resp.get("model_display", resp["model"]),
                "probe": resp["probe"],
                "dimension": resp["dimension"],
                "disease": resp["disease"],
                "disease_display": resp["disease_display"],
                "category": categories.get(resp["disease"], "Other"),
                "semantic_similarity": round(float(top_k_scores[j]), 6),
                "n_abstracts": abstract_emb.shape[0],
                "total_words": len(resp["response"].split()),
            }

        diseases_processed += 1
        if diseases_processed % 25 == 0:
            print(f"    {diseases_processed}/{len(disease_groups)} diseases scored "
                  f"({disease}: {abstract_emb.shape[0]} abstracts)")

        del abstract_emb  # Free memory

    # Remove None entries (diseases with no embeddings)
    all_scores = [s for s in all_scores if s is not None and s["semantic_similarity"] is not None]
    print(f"\n  Total scored: {len(all_scores)} responses")

    # Save
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORES_DIR / "all_scores.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"  Saved: {SCORES_DIR / 'all_scores.json'}")

    generate_summaries(all_scores)
    print_highlights(all_scores)


def generate_summaries(all_scores):
    """Generate aggregated summary CSVs."""

    # --- Per-disease ---
    disease_map = {}
    for s in all_scores:
        d = s["disease"]
        if d not in disease_map:
            disease_map[d] = {"sim": [], "cat": s["category"],
                              "n_abs": s["n_abstracts"]}
        disease_map[d]["sim"].append(s["semantic_similarity"])

    with open(SCORES_DIR / "disease_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["disease", "category", "mean_semantic_similarity",
                     "std_semantic_similarity", "n_abstracts", "n_scores"])
        for d in sorted(disease_map.keys()):
            v = disease_map[d]
            w.writerow([
                d, v["cat"],
                f"{stats_mod.mean(v['sim']):.6f}",
                f"{stats_mod.stdev(v['sim']):.6f}" if len(v["sim"]) > 1 else "0",
                v["n_abs"], len(v["sim"]),
            ])
    print(f"  Saved: {SCORES_DIR / 'disease_scores.csv'}")

    # --- Per-model ---
    model_map = {}
    for s in all_scores:
        m = s["model"]
        if m not in model_map:
            model_map[m] = {"sim": [], "display": s["model_display"]}
        model_map[m]["sim"].append(s["semantic_similarity"])

    with open(SCORES_DIR / "model_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "display_name", "mean_semantic_similarity",
                     "std_semantic_similarity", "n_scores"])
        for m in sorted(model_map.keys()):
            v = model_map[m]
            w.writerow([
                m, v["display"],
                f"{stats_mod.mean(v['sim']):.6f}",
                f"{stats_mod.stdev(v['sim']):.6f}" if len(v["sim"]) > 1 else "0",
                len(v["sim"]),
            ])
    print(f"  Saved: {SCORES_DIR / 'model_scores.csv'}")

    # --- Per-probe ---
    probe_map = {}
    for s in all_scores:
        p = s["probe"]
        if p not in probe_map:
            probe_map[p] = {"sim": [], "dim": s["dimension"]}
        probe_map[p]["sim"].append(s["semantic_similarity"])

    with open(SCORES_DIR / "probe_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe", "dimension", "mean_semantic_similarity",
                     "std_semantic_similarity", "n_scores"])
        for p in sorted(probe_map.keys()):
            v = probe_map[p]
            w.writerow([
                p, v["dim"],
                f"{stats_mod.mean(v['sim']):.6f}",
                f"{stats_mod.stdev(v['sim']):.6f}" if len(v["sim"]) > 1 else "0",
                len(v["sim"]),
            ])
    print(f"  Saved: {SCORES_DIR / 'probe_scores.csv'}")


def print_highlights(all_scores):
    """Print key findings."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Model rankings
    model_scores = {}
    for s in all_scores:
        m = s["model_display"]
        if m not in model_scores:
            model_scores[m] = []
        model_scores[m].append(s["semantic_similarity"])

    print(f"\nModel Rankings (mean similarity to top-{TOP_K} abstracts):")
    ranked = sorted(model_scores.items(),
                    key=lambda x: stats_mod.mean(x[1]), reverse=True)
    for i, (model, sims) in enumerate(ranked, 1):
        print(f"  {i}. {model:<25s} {stats_mod.mean(sims):.6f} "
              f"(sd={stats_mod.stdev(sims):.4f}, n={len(sims)})")

    # Category rankings
    cat_scores = {}
    for s in all_scores:
        c = s["category"]
        if c not in cat_scores:
            cat_scores[c] = []
        cat_scores[c].append(s["semantic_similarity"])

    print(f"\nCategory Rankings (mean similarity to top-{TOP_K} abstracts):")
    ranked_cats = sorted(cat_scores.items(),
                         key=lambda x: stats_mod.mean(x[1]), reverse=True)
    for cat, sims in ranked_cats:
        print(f"  {cat:<20s} {stats_mod.mean(sims):.6f} (n={len(sims)})")

    # NTD vs non-NTD
    ntd = [s["semantic_similarity"] for s in all_scores if s["category"] == "NTD"]
    non_ntd = [s["semantic_similarity"] for s in all_scores if s["category"] != "NTD"]
    if ntd and non_ntd:
        print(f"\n  NTD mean:     {stats_mod.mean(ntd):.6f} (n={len(ntd)})")
        print(f"  Non-NTD mean: {stats_mod.mean(non_ntd):.6f} (n={len(non_ntd)})")

    # Probe rankings
    probe_scores = {}
    for s in all_scores:
        p = s["probe"]
        if p not in probe_scores:
            probe_scores[p] = []
        probe_scores[p].append(s["semantic_similarity"])

    print(f"\nProbe Rankings (mean similarity to top-{TOP_K} abstracts):")
    ranked_probes = sorted(probe_scores.items(),
                           key=lambda x: stats_mod.mean(x[1]), reverse=True)
    for probe, sims in ranked_probes:
        print(f"  {probe:<25s} {stats_mod.mean(sims):.6f}")


if __name__ == "__main__":
    main()
