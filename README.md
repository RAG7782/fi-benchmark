# Framework Injection Benchmark Suite

## Components

- `fi_generator.py` — Generates injectable frameworks for any domain
- `benchmark.py` — Full PE vs CE vs FI benchmark (10 domains x 3 conditions x N runs)
- `validate.py` — Quick validation (3 domains x 1 run)

## Usage

```bash
# Generate a framework
python fi_generator.py --domain "corporate law M&A due diligence"

# Quick validation (9 API calls)
python validate.py

# Full benchmark (90+ API calls)
python benchmark.py --domains all --runs 3
```

## Methodology

Three conditions compared:
- **PE** (Prompt Engineering): simple task instruction
- **CE** (Context Engineering): task + expert context + structure
- **FI** (Framework Injection): complete 5-type injectable framework + task

Evaluation: LLM-as-judge on 6 criteria (1-5 scale):
1. Domain Accuracy
2. Reasoning Depth
3. Completeness
4. Actionability
5. Hallucination Check
6. Professional Tone

## Reference

Gomes, R. A. (2026). From Commands to Cognition: Digital Craftsmanship and the Framework Injection Paradigm. DOI: 10.5281/zenodo.19344789
