# The Bias Amplification Cascade

**Code and data for:** "The Bias Amplification Cascade: How Structural Inequities in Biomedical Knowledge Propagate Through Large Language Models"

Manuel Corpas, Allan Kalungi, Sainabou Ndure, Heinner Guio, Oyesola Ojewunmi, Segun Fatumo

*Communications Biology* (2026)

## Overview

This repository contains the benchmark pipeline and analysis code for evaluating six frontier LLMs across 175 Global Burden of Disease categories in ten biomedical knowledge domains (10,500 total queries). The pipeline measures semantic alignment between LLM responses and the published PubMed literature using PubMedBERT embeddings.

## Repository Structure

```
PYTHON/                          # Analysis pipeline (ordered scripts)
  00-build-ground-truth.py       # Extract PubMed abstracts, compute BERT centroids
  01-extract-heim-diseases.py    # Load HEIM metrics for 175 GBD diseases
  02-collect-responses.py        # Query 6 LLMs via API (10,500 queries)
  02a-collect-missing.py         # Retry failed LLM queries
  03-evaluate-responses.py       # PubMedBERT embedding + cosine similarity scoring
  04-correlate-sii-performance.py # SII correlation analysis + regression
  05-benchmark-summary.py        # Aggregate results and rankings
  06-generate-figures.py         # Generate Figures 1-5 (PDF + PNG)

DATA/                            # Input data
  heim_diseases.csv              # 175 GBD diseases with HEIM metrics (SII, KTP, RCC)
  ground-truth/
    disease_ground_truth.csv     # Disease categories and ground-truth metadata

RESULTS/                         # Pipeline outputs
  sii_correlation_results.json   # H1 + H2 statistical tests (source of truth)
  sii_correlation_results.csv    # Per-disease merged data for plotting
  benchmark_summary.json         # Model and probe rankings
  SCORES/
    all_scores.json              # Individual semantic similarity scores (n=10,453)
    disease_scores.csv           # Mean score per disease
```

## Pipeline

Scripts are numbered in execution order:

1. **00-build-ground-truth.py** — Retrieves top-50 PubMed abstracts per disease via NCBI E-utilities; embeds with PubMedBERT (768 dimensions); computes centroid per disease.

2. **01-extract-heim-diseases.py** — Loads HEIM framework metrics (Semantic Isolation Index, Knowledge Trajectory Profile, Research Completeness Category) for 175 GBD 2021 disease categories.

3. **02-collect-responses.py** — Queries six frontier LLMs (Gemini 3 Pro, Claude Opus 4.5, Claude Sonnet 4, GPT-5.2, Mistral Large, DeepSeek V3) with 10 standardised probe templates per disease.

4. **03-evaluate-responses.py** — Embeds LLM responses with PubMedBERT; computes cosine similarity to ground-truth centroids.

5. **04-correlate-sii-performance.py** — Tests H1 (SII-performance correlation) and H2 (partial correlation controlling for publication volume); fits multiple regression model.

6. **05-benchmark-summary.py** — Aggregates scores into model rankings, probe rankings, and summary statistics.

7. **06-generate-figures.py** — Generates all five publication figures (PDF + PNG at 600 DPI).

## Requirements

```
pip install -r requirements.txt
```

API keys are required in a `.env` file for steps 02/02a (LLM queries):
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `MISTRAL_API_KEY`
- `DEEPSEEK_API_KEY`

Steps 03 onwards can be run using only the pre-computed data in `RESULTS/`.

## Key Results

- **H1**: Semantic isolation correlates with LLM performance (Pearson r = −0.243, p = 0.001; Spearman ρ = −0.275, p = 0.0002)
- **H2**: Partial correlation controlling for publication volume (r = 0.331, p = 7.9 × 10⁻⁶)
- **Multiple regression**: R² = 0.666 (F(4,170) = 84.9, p < 10⁻¹⁶)
- **Most affected domain**: Diagnostics (r = −0.313)

## Data Availability

- **HEIM metrics**: Provided in `DATA/heim_diseases.csv` (175 diseases × SII, KTP, RCC, publication counts)
- **LLM benchmark scores**: Provided in `RESULTS/SCORES/all_scores.json` (10,453 individual similarity scores)
- **Statistical results**: Provided in `RESULTS/sii_correlation_results.json`
- **PubMed ground truth**: Reproducible via `00-build-ground-truth.py` using NCBI E-utilities (public API)

Raw LLM responses (10,500 JSON files) are not included due to size but are fully reproducible by running `02-collect-responses.py` with valid API keys.

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.
