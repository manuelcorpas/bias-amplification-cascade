#!/usr/bin/env python3
"""
Collect LLM Responses — African Health Benchmark
=================================================
Queries 6 frontier LLMs with 10 probes × 175 diseases = 10,500 queries.
Saves raw responses for subsequent evaluation.

This follows the same evaluation-of-commercial-tools framing as
Sorin et al. (2025), who benchmarked 9 LLMs across ~500K experiments.

Models (6 frontier LLMs, February 2026):
    - Gemini 3 Pro (Google)
    - Claude Opus 4.5 (Anthropic)
    - Claude Sonnet 4 (Anthropic)
    - GPT-5.2 (OpenAI)
    - Mistral Large (Mistral)
    - DeepSeek V3 (DeepSeek)

Environment variables required:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
    MISTRAL_API_KEY, DEEPSEEK_API_KEY

Usage:
    python 02-collect-responses.py [--models MODEL1,MODEL2] [--probes P1,P2]
                                    [--resume] [--dry-run]

Author: Manuel Corpas
Date: February 2026
"""

import os
import csv
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded API keys from {env_path}")
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

# ---------------------------------------------------------------------------
# Probe definitions — 10 knowledge dimensions
# ---------------------------------------------------------------------------
PROBES = {
    "P01_burden": {
        "template": (
            "What is the current disease burden of {DISEASE} in sub-Saharan "
            "Africa, including DALYs, mortality, and affected populations?"
        ),
        "dimension": "Epidemiological",
    },
    "P02_etiology": {
        "template": (
            "What are the primary causes and risk factors for {DISEASE} in "
            "African populations, including environmental, genetic, and social "
            "determinants?"
        ),
        "dimension": "Biomedical",
    },
    "P03_genomics": {
        "template": (
            "What is known about the genetic and genomic basis of {DISEASE} "
            "in African populations, including any population-specific variants "
            "or pharmacogenomic considerations?"
        ),
        "dimension": "Foundational Science",
    },
    "P04_diagnostics": {
        "template": (
            "What are the current approaches to diagnosing {DISEASE} in "
            "African healthcare settings, including point-of-care and "
            "resource-limited options?"
        ),
        "dimension": "Clinical",
    },
    "P05_treatment": {
        "template": (
            "What are the most effective treatments for {DISEASE} available "
            "in African health systems, and what are the key barriers to access?"
        ),
        "dimension": "Clinical",
    },
    "P06_prevention": {
        "template": (
            "What are the most effective public health interventions and "
            "prevention strategies for {DISEASE} in African settings?"
        ),
        "dimension": "Public Health",
    },
    "P07_research": {
        "template": (
            "What are the most important research findings on {DISEASE} in "
            "African populations in the last 10 years, including key studies "
            "and authors?"
        ),
        "dimension": "Literature Knowledge",
    },
    "P08_health_systems": {
        "template": (
            "How do African health systems currently manage {DISEASE}, "
            "including workforce capacity, supply chains, and referral pathways?"
        ),
        "dimension": "Health Systems",
    },
    "P09_equity": {
        "template": (
            "What are the key health equity challenges related to {DISEASE} "
            "in Africa, including disparities by gender, geography, "
            "socioeconomic status, and age?"
        ),
        "dimension": "Equity",
    },
    "P10_policy": {
        "template": (
            "What are the current national and regional policies addressing "
            "{DISEASE} in Africa, and what policy gaps remain?"
        ),
        "dimension": "Governance",
    },
}

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODELS = {
    "Gemini_3_Pro": {
        "provider": "google",
        "model_id": "gemini-3-pro-preview",
        "display_name": "Gemini 3 Pro",
    },
    "Claude_Opus_4.5": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",
        "display_name": "Claude Opus 4.5",
    },
    "Claude_Sonnet_4": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
    },
    "GPT-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",
        "display_name": "GPT-5.2",
    },
    "Mistral_Large": {
        "provider": "mistral",
        "model_id": "mistral-large-latest",
        "display_name": "Mistral Large",
    },
    "DeepSeek_V3": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "display_name": "DeepSeek V3",
    },
}

# ---------------------------------------------------------------------------
# API query functions (adapted from PLOS codebase)
# ---------------------------------------------------------------------------

def query_anthropic_api(model_id: str, prompt: str) -> str:
    if anthropic is None:
        return "[ERROR: anthropic package not installed]"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR: ANTHROPIC_API_KEY not set]"
    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model=model_id,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"[ERROR: {e}]"


def query_openai_api(model_id: str, prompt: str) -> str:
    if openai is None:
        return "[ERROR: openai package not installed]"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "[ERROR: OPENAI_API_KEY not set]"
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {e}]"


def query_google_api(model_id: str, prompt: str) -> str:
    if genai is None:
        return "[ERROR: google-generativeai package not installed]"
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "[ERROR: GOOGLE_API_KEY not set]"
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[ERROR: {e}]"


def query_mistral_api(model_id: str, prompt: str) -> str:
    if Mistral is None:
        return "[ERROR: mistralai package not installed]"
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return "[ERROR: MISTRAL_API_KEY not set]"
    client = Mistral(api_key=api_key)
    try:
        response = client.chat.complete(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {e}]"


def query_deepseek_api(model_id: str, prompt: str) -> str:
    if openai is None:
        return "[ERROR: openai package not installed]"
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "[ERROR: DEEPSEEK_API_KEY not set]"
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {e}]"


PROVIDER_FN = {
    "anthropic": query_anthropic_api,
    "openai": query_openai_api,
    "google": query_google_api,
    "mistral": query_mistral_api,
    "deepseek": query_deepseek_api,
}


def query_model(model_key: str, prompt: str) -> str:
    config = MODELS[model_key]
    fn = PROVIDER_FN[config["provider"]]
    return fn(config["model_id"], prompt)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def load_diseases():
    """Load disease list from extracted HEIM data."""
    diseases_file = Path(__file__).parent.parent / "DATA" / "heim_diseases.csv"
    if not diseases_file.exists():
        print(f"ERROR: {diseases_file} not found. Run 01-extract-heim-diseases.py first.")
        raise SystemExit(1)
    diseases = []
    with open(diseases_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            diseases.append(row["disease"])
    return diseases


def get_completed_queries(output_dir: Path):
    """Scan output dir for already-completed queries (for resume support)."""
    completed = set()
    for fpath in output_dir.glob("*.json"):
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            completed.add((data["model"], data["probe"], data["disease"]))
        except Exception:
            pass
    return completed


def main():
    parser = argparse.ArgumentParser(description="Collect LLM responses for African health benchmark")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (default: all)")
    parser.add_argument("--probes", type=str, default=None,
                        help="Comma-separated probe keys (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-collected queries")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print queries without executing")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls in seconds")
    args = parser.parse_args()

    # Select models and probes
    model_keys = args.models.split(",") if args.models else list(MODELS.keys())
    probe_keys = args.probes.split(",") if args.probes else list(PROBES.keys())

    diseases = load_diseases()
    total = len(model_keys) * len(probe_keys) * len(diseases)

    print("=" * 70)
    print("African Health LLM Benchmark — Response Collection")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(model_keys)}  Probes: {len(probe_keys)}  Diseases: {len(diseases)}")
    print(f"Total queries: {total}")
    print("=" * 70)

    # Output directory
    output_dir = Path(__file__).parent.parent / "RESULTS" / "RESPONSES"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed = get_completed_queries(output_dir) if args.resume else set()
    if completed:
        print(f"Resuming: {len(completed)} queries already completed")

    # Check API keys
    print("\nAPI Key Status:")
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                "MISTRAL_API_KEY", "DEEPSEEK_API_KEY"]:
        status = "SET" if os.environ.get(key) else "NOT SET"
        print(f"  {key}: {status}")

    # Collect — parallelised across models for each (disease, probe) pair
    count = 0
    errors = 0
    skipped = 0
    saved = 0

    # Build work items: iterate disease × probe, query all models in parallel
    work_items = []
    for probe_key in probe_keys:
        for disease in diseases:
            for model_key in model_keys:
                count += 1
                if (model_key, probe_key, disease) in completed:
                    skipped += 1
                    continue
                work_items.append((model_key, probe_key, disease))

    if args.dry_run:
        print(f"\n[DRY RUN] {len(work_items)} queries to execute ({skipped} skipped)")
        for i, (mk, pk, dis) in enumerate(work_items[:3]):
            disease_display = dis.replace("_", " ")
            prompt = PROBES[pk]["template"].format(DISEASE=disease_display)
            print(f"\n[DRY RUN] {MODELS[mk]['display_name']} | {pk} | {disease_display}")
            print(f"  Prompt: {prompt[:100]}...")
        if len(work_items) > 3:
            print(f"\n  ... ({len(work_items) - 3} more queries)")
        print(f"\n{'=' * 70}")
        print("COLLECTION COMPLETE")
        print(f"  Total queries:   {count}")
        print(f"  Skipped (resume): {skipped}")
        print(f"  Errors:          0")
        print(f"  Output dir:      {output_dir}")
        print(f"{'=' * 70}")
        return

    print(f"\n  {len(work_items)} queries to execute ({skipped} already done)")

    def execute_query(item):
        model_key, probe_key, disease = item
        model_name = MODELS[model_key]["display_name"]
        probe = PROBES[probe_key]
        disease_display = disease.replace("_", " ")
        prompt = probe["template"].format(DISEASE=disease_display)

        response = query_model(model_key, prompt)

        is_error = response.startswith("[ERROR")

        result = {
            "model": model_key,
            "model_display": model_name,
            "probe": probe_key,
            "dimension": probe["dimension"],
            "disease": disease,
            "disease_display": disease_display,
            "prompt": prompt,
            "response": response,
            "is_error": is_error,
            "timestamp": datetime.now().isoformat(),
            "char_count": len(response),
        }

        fname = f"{model_key}__{probe_key}__{disease}.json"
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return is_error, model_name, probe_key, disease_display

    # Use ThreadPoolExecutor — 6 threads (one per model) for I/O-bound API calls
    max_workers = min(12, len(model_keys) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute_query, item): item for item in work_items}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                is_error, model_name, probe_key, disease_display = future.result()
                if is_error:
                    errors += 1
                saved += 1
            except Exception as e:
                errors += 1
                item = futures[future]
                print(f"\n  EXCEPTION: {item[0]} | {item[1]} | {item[2]}: {e}")

            if done_count % 50 == 0 or done_count == len(work_items):
                print(f"  [{done_count}/{len(work_items)}] completed "
                      f"({errors} errors)", flush=True)

    print(f"\n\n{'=' * 70}")
    print("COLLECTION COMPLETE")
    print(f"  Total queries:   {count}")
    print(f"  Skipped (resume): {skipped}")
    print(f"  Errors:          {errors}")
    print(f"  Output dir:      {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
