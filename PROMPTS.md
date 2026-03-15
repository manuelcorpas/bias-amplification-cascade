# LLM Probe Templates

Ten standardised probes spanning the biomedical knowledge lifecycle were used to query each of the six frontier LLMs across 175 GBD disease categories (10,500 total queries). All queries used default model parameters with no system prompts.

The `{DISEASE}` placeholder was replaced with each GBD disease name.

## Probes

| ID | Domain | Template |
|----|--------|----------|
| P01 | Epidemiological | What is the current disease burden of {DISEASE} in sub-Saharan Africa, including DALYs, mortality, and affected populations? |
| P02 | Biomedical | What are the primary causes and risk factors for {DISEASE} in African populations, including environmental, genetic, and social determinants? |
| P03 | Foundational Science | What is known about the genetic and genomic basis of {DISEASE} in African populations, including any population-specific variants or pharmacogenomic considerations? |
| P04 | Clinical | What are the current approaches to diagnosing {DISEASE} in African healthcare settings, including point-of-care and resource-limited options? |
| P05 | Clinical | What are the most effective treatments for {DISEASE} available in African health systems, and what are the key barriers to access? |
| P06 | Public Health | What are the most effective public health interventions and prevention strategies for {DISEASE} in African settings? |
| P07 | Literature Knowledge | What are the most important research findings on {DISEASE} in African populations in the last 10 years, including key studies and authors? |
| P08 | Health Systems | How do African health systems currently manage {DISEASE}, including workforce capacity, supply chains, and referral pathways? |
| P09 | Equity | What are the key health equity challenges related to {DISEASE} in Africa, including disparities by gender, geography, socioeconomic status, and age? |
| P10 | Governance | What are the current national and regional policies addressing {DISEASE} in Africa, and what policy gaps remain? |

## Models Evaluated

| Model | Provider | Model ID |
|-------|----------|----------|
| Gemini 3 Pro | Google | gemini-3-pro-preview |
| Claude Opus 4.5 | Anthropic | claude-opus-4-5-20251101 |
| Claude Sonnet 4 | Anthropic | claude-sonnet-4-20250514 |
| GPT-5.2 | OpenAI | gpt-5.2 |
| Mistral Large | Mistral AI | mistral-large-latest |
| DeepSeek V3 | DeepSeek | deepseek-chat |

## Query Parameters

- Temperature: default (not specified)
- System prompt: none
- Max tokens: 4096
- No few-shot examples
- Single query per disease-probe-model combination
