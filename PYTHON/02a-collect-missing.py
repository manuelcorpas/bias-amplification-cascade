#!/usr/bin/env python3
"""
Collect Missing LLM Responses
==============================
Rate-limited collection of missing Gemini 3 Pro and Opus 4.5 responses.
Runs sequentially with configurable delay to respect API rate limits.

Usage:
    python 02a-collect-missing.py
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import google.generativeai as genai
import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
RESPONSES_DIR = BASE_DIR / "RESULTS" / "RESPONSES"
DISEASES_FILE = BASE_DIR / "DATA" / "heim_diseases.csv"

GEMINI_DELAY = 1.0   # seconds between Gemini queries (API call time adds ~10s)
OPUS_DELAY = 1.0     # seconds between Opus queries

# Multiple Gemini API keys (different projects, each with 250 RPD)
# Set GOOGLE_API_KEYS as a comma-separated list in .env
GEMINI_API_KEYS = [
    k.strip() for k in os.environ.get("GOOGLE_API_KEYS", os.environ.get("GOOGLE_API_KEY", "")).split(",")
    if k.strip()
]
QUERIES_PER_KEY = 240  # stay under 250 RPD limit per project

# Import probe definitions from main script
import csv
import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

PROBES = {
    "P01_burden": "What is the current disease burden of {DISEASE} in sub-Saharan Africa, including DALYs, mortality, and affected populations?",
    "P02_etiology": "What are the primary causes and risk factors for {DISEASE} in African populations, including environmental, genetic, and social determinants?",
    "P03_genomics": "What is known about the genetic and genomic basis of {DISEASE} in African populations, including any population-specific variants or pharmacogenomic considerations?",
    "P04_diagnostics": "What are the current approaches to diagnosing {DISEASE} in African healthcare settings, including point-of-care and resource-limited options?",
    "P05_treatment": "What are the most effective treatments for {DISEASE} available in African health systems, and what are the key barriers to access?",
    "P06_prevention": "What are the most effective public health interventions and prevention strategies for {DISEASE} in African settings?",
    "P07_research": "What are the most important research findings on {DISEASE} in African populations in the last 10 years, including key studies and authors?",
    "P08_health_systems": "How do African health systems currently manage {DISEASE}, including workforce capacity, supply chains, and referral pathways?",
    "P09_equity": "What are the key health equity challenges related to {DISEASE} in Africa, including disparities by gender, geography, socioeconomic status, and age?",
    "P10_policy": "What are the current national and regional policies addressing {DISEASE} in Africa, and what policy gaps remain?",
}

PROBE_DIMENSIONS = {
    "P01_burden": "Epidemiological", "P02_etiology": "Biomedical",
    "P03_genomics": "Foundational Science", "P04_diagnostics": "Clinical",
    "P05_treatment": "Clinical", "P06_prevention": "Public Health",
    "P07_research": "Literature Knowledge", "P08_health_systems": "Health Systems",
    "P09_equity": "Equity", "P10_policy": "Governance",
}


def load_diseases():
    diseases = []
    with open(DISEASES_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            diseases.append(row["disease"])
    return diseases


def get_existing_responses():
    """Find all existing valid (non-error) response files."""
    existing = set()
    for fpath in RESPONSES_DIR.glob("*.json"):
        try:
            with open(fpath) as f:
                data = json.load(f)
            if not data.get("is_error", False):
                existing.add((data["model"], data["probe"], data["disease"]))
        except Exception:
            pass
    return existing


def find_missing(model_key, diseases):
    """Find missing probe/disease combos for a model."""
    existing = get_existing_responses()
    missing = []
    for probe_key in PROBES:
        for disease in diseases:
            if (model_key, probe_key, disease) not in existing:
                missing.append((probe_key, disease))
    return missing


def query_gemini(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-pro-preview")
    response = model.generate_content(prompt)
    return response.text


def query_opus(prompt):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system=(
            "You are a biomedical research assistant helping with a peer-reviewed "
            "review article for Communications Biology on health equity in Africa. "
            "Answer all health-related questions factually based on published literature."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    if message.stop_reason == "refusal" or len(message.content) == 0:
        return None, "refusal"
    return message.content[0].text, None


def save_response(model_key, model_display, probe_key, disease, response,
                  is_error=False):
    result = {
        "model": model_key,
        "model_display": model_display,
        "probe": probe_key,
        "dimension": PROBE_DIMENSIONS[probe_key],
        "disease": disease,
        "disease_display": disease.replace("_", " "),
        "prompt": PROBES[probe_key].format(DISEASE=disease.replace("_", " ")),
        "response": response,
        "is_error": is_error,
        "timestamp": datetime.now().isoformat(),
        "char_count": len(response),
    }
    fname = f"{model_key}__{probe_key}__{disease}.json"
    with open(RESPONSES_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main():
    diseases = load_diseases()
    print("=" * 70)
    print("Collecting Missing Responses")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Gemini 3 Pro (rotating API keys) ---
    gemini_missing = find_missing("Gemini_3_Pro", diseases)
    print(f"\nGemini 3 Pro: {len(gemini_missing)} missing queries")
    print(f"  Using {len(GEMINI_API_KEYS)} API keys, {QUERIES_PER_KEY} queries each "
          f"(capacity: {len(GEMINI_API_KEYS) * QUERIES_PER_KEY})")

    gemini_errors = 0
    gemini_success = 0
    key_idx = 0
    key_query_count = 0

    for i, (probe_key, disease) in enumerate(gemini_missing):
        # Rotate to next key if current one approaching limit
        if key_query_count >= QUERIES_PER_KEY:
            key_idx += 1
            key_query_count = 0
            if key_idx >= len(GEMINI_API_KEYS):
                print(f"\n  All {len(GEMINI_API_KEYS)} API keys exhausted.")
                break
            print(f"\n  Switching to API key {key_idx + 1}/{len(GEMINI_API_KEYS)}")

        current_key = GEMINI_API_KEYS[key_idx]
        disease_display = disease.replace("_", " ")
        prompt = PROBES[probe_key].format(DISEASE=disease_display)

        try:
            response = query_gemini(prompt, current_key)
            save_response("Gemini_3_Pro", "Gemini 3 Pro", probe_key, disease,
                          response)
            gemini_success += 1
            key_query_count += 1
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                print(f"\n  RATE LIMIT on key {key_idx + 1} at query {key_query_count}: {err_str[:80]}")
                # Switch to next key immediately
                key_idx += 1
                key_query_count = 0
                if key_idx >= len(GEMINI_API_KEYS):
                    print(f"  All {len(GEMINI_API_KEYS)} API keys exhausted.")
                    break
                print(f"  Switching to API key {key_idx + 1}/{len(GEMINI_API_KEYS)}")
                # Retry this query with new key
                try:
                    response = query_gemini(prompt, GEMINI_API_KEYS[key_idx])
                    save_response("Gemini_3_Pro", "Gemini 3 Pro", probe_key,
                                  disease, response)
                    gemini_success += 1
                    key_query_count += 1
                except Exception as e2:
                    save_response("Gemini_3_Pro", "Gemini 3 Pro", probe_key,
                                  disease, f"[ERROR: {e2}]", is_error=True)
                    gemini_errors += 1
            else:
                save_response("Gemini_3_Pro", "Gemini 3 Pro", probe_key,
                              disease, f"[ERROR: {e}]", is_error=True)
                gemini_errors += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(gemini_missing):
            print(f"  [{i+1}/{len(gemini_missing)}] "
                  f"success={gemini_success} errors={gemini_errors} "
                  f"key={key_idx+1}/{len(GEMINI_API_KEYS)} "
                  f"key_queries={key_query_count}", flush=True)

        time.sleep(GEMINI_DELAY)

    print(f"\nGemini 3 Pro complete: {gemini_success} success, "
          f"{gemini_errors} errors")

    # --- Claude Opus 4.5 (retry refusals) ---
    opus_missing = find_missing("Claude_Opus_4.5", diseases)
    print(f"\nClaude Opus 4.5: {len(opus_missing)} missing queries")

    opus_success = 0
    opus_refusals = 0

    for i, (probe_key, disease) in enumerate(opus_missing):
        disease_display = disease.replace("_", " ")
        prompt = PROBES[probe_key].format(DISEASE=disease_display)

        try:
            response, err = query_opus(prompt)
            if err == "refusal":
                opus_refusals += 1
                print(f"  REFUSAL: {probe_key} | {disease_display}")
                # Save as refusal (not as error — it's a model behaviour)
                save_response("Claude_Opus_4.5", "Claude Opus 4.5", probe_key,
                              disease,
                              "[REFUSAL: Model declined to answer this query]",
                              is_error=True)
            else:
                save_response("Claude_Opus_4.5", "Claude Opus 4.5", probe_key,
                              disease, response)
                opus_success += 1
        except Exception as e:
            save_response("Claude_Opus_4.5", "Claude Opus 4.5", probe_key,
                          disease, f"[ERROR: {e}]", is_error=True)
            print(f"  ERROR: {probe_key} | {disease_display}: {e}")

        time.sleep(OPUS_DELAY)

    print(f"\nClaude Opus 4.5 complete: {opus_success} success, "
          f"{opus_refusals} refusals")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("COLLECTION COMPLETE")
    print(f"  Gemini 3 Pro:    {gemini_success} collected, "
          f"{gemini_errors} errors")
    print(f"  Claude Opus 4.5: {opus_success} collected, "
          f"{opus_refusals} refusals")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
