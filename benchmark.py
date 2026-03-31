"""
PE vs CE vs FI Benchmark Suite v1.0

Compares Prompt Engineering, Context Engineering, and Framework Injection
across multiple domains with blind evaluation.

Usage:
    python benchmark.py --domains all --output results/
    python benchmark.py --domains "legal,medical" --runs 3
"""

import anthropic
import json
import os
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============================================================
# BENCHMARK DOMAINS — 10 diverse domains
# ============================================================

DOMAINS = {
    "legal_contract": {
        "name": "Corporate Contract Analysis",
        "task": "Analyze the following contract clause for risks, ambiguities, and compliance issues. Identify specific legal concerns and propose alternative language.",
        "test_input": "The Supplier shall indemnify the Client against all losses arising from the Supplier's negligence, provided that the Client notifies the Supplier within 30 days of becoming aware of such loss. The Supplier's total liability under this agreement shall not exceed the fees paid in the 12 months preceding the claim. This limitation shall not apply to losses arising from willful misconduct or gross negligence.",
        "expert_criteria": ["identifies indemnification scope", "flags notification period risk", "analyzes liability cap", "distinguishes negligence types", "proposes specific alternatives", "cites relevant legal principles"]
    },
    "medical_diagnosis": {
        "name": "Clinical Differential Diagnosis",
        "task": "A 45-year-old male presents with acute chest pain radiating to the left arm, diaphoresis, and shortness of breath. BP 150/90, HR 110, SpO2 94%. ECG shows ST elevation in leads II, III, aVF. Provide differential diagnosis with reasoning.",
        "test_input": "Patient history: smoker 20 pack-years, family history of CAD (father MI at 52), BMI 31, type 2 diabetes diagnosed 3 years ago, on metformin. No prior cardiac history. Pain started 2 hours ago while at rest.",
        "expert_criteria": ["identifies STEMI as primary", "considers differential (PE, dissection, pericarditis)", "integrates risk factors", "recommends immediate interventions", "follows systematic approach", "addresses time-sensitivity"]
    },
    "financial_analysis": {
        "name": "Financial Due Diligence",
        "task": "Evaluate the following financial metrics for a potential acquisition target and identify red flags.",
        "test_input": "Revenue: $50M (up 40% YoY), EBITDA margin: 22% (industry avg 15%), Customer concentration: top client = 35% revenue, Receivables DSO: 85 days (industry avg 45), Inventory turnover: 3.2x (industry avg 6.1x), Debt/EBITDA: 4.2x, Free cash flow: negative last 2 quarters despite profit growth.",
        "expert_criteria": ["flags customer concentration risk", "identifies DSO concern", "analyzes inventory turnover gap", "questions revenue quality", "assesses leverage risk", "examines FCF vs profit divergence"]
    },
    "cybersecurity": {
        "name": "Security Incident Analysis",
        "task": "Analyze the following security alert and recommend response actions.",
        "test_input": "Alert: Unusual outbound traffic detected from internal server 10.0.1.45 to IP 185.234.x.x (known C2 infrastructure) at 03:00 AM local time. Volume: 2.3GB over 4 hours. Server role: HR database containing PII of 15,000 employees. Last patched: 45 days ago. No authorized maintenance window. EDR shows powershell execution with base64 encoded commands.",
        "expert_criteria": ["identifies likely compromise", "recommends containment steps", "addresses data breach implications", "suggests forensic preservation", "considers regulatory notification", "prioritizes actions by urgency"]
    },
    "architecture_review": {
        "name": "Software Architecture Review",
        "task": "Review the following system architecture and identify scalability, reliability, and security concerns.",
        "test_input": "Monolithic Python/Django application serving 50K concurrent users. Single PostgreSQL database (16 cores, 64GB RAM) handling all reads/writes. Session state stored in database. File uploads stored on local disk. Authentication via custom JWT implementation. No rate limiting. Deployed on single cloud region. Background jobs processed synchronously in request handlers. Average response time: 800ms, p99: 12s.",
        "expert_criteria": ["identifies single points of failure", "recommends database scaling strategy", "addresses session management", "flags security concerns", "proposes async processing", "suggests multi-region strategy"]
    },
    "tax_planning": {
        "name": "Brazilian Tax Planning",
        "task": "Analyze the tax implications of the following business restructuring for a Brazilian company.",
        "test_input": "Holding company (Lucro Real) with 3 subsidiaries: manufacturing (Lucro Real, R$80M revenue), services (Lucro Presumido, R$15M revenue), e-commerce (Simples Nacional, R$4M revenue). Owner wants to: (1) merge services into manufacturing, (2) distribute R$5M in dividends, (3) transfer intellectual property to a new offshore entity. Context: post-Reform (IBS/CBS operational since Jan 2026).",
        "expert_criteria": ["analyzes each restructuring step", "considers reform implications", "identifies tax risks", "addresses transfer pricing", "evaluates dividend taxation", "proposes compliant alternatives"]
    },
    "education_design": {
        "name": "Instructional Design for Complex Topic",
        "task": "Design a learning sequence for teaching the following complex topic to graduate students.",
        "test_input": "Topic: Persistent Homology in Topological Data Analysis. Audience: Computer Science graduate students with linear algebra but no topology background. Duration: 4 sessions of 90 minutes. Goal: students should be able to compute persistence diagrams from point cloud data and interpret them for practical applications.",
        "expert_criteria": ["scaffolds prerequisites", "sequences concepts logically", "includes hands-on exercises", "connects to applications", "addresses common misconceptions", "designs assessments"]
    },
    "environmental_impact": {
        "name": "Environmental Impact Assessment",
        "task": "Evaluate the environmental implications of the following industrial project.",
        "test_input": "Proposed lithium extraction facility in the Atacama region (Chile). Open-pit mining with evaporation ponds. Expected production: 25,000 tonnes LCE/year. Water usage: 2 million cubic meters/year from local aquifer. Area: 500 hectares of desert ecosystem with endemic species. Indigenous communities within 20km. Duration: 25-year concession.",
        "expert_criteria": ["assesses water scarcity impact", "evaluates ecosystem disruption", "considers indigenous rights", "analyzes lifecycle emissions", "proposes mitigation measures", "addresses cumulative effects"]
    },
    "product_strategy": {
        "name": "Product Strategy Analysis",
        "task": "Evaluate the following product strategy and identify risks and opportunities.",
        "test_input": "B2B SaaS company (ARR $8M, 200 customers, NRR 95%) plans to: (1) launch AI features requiring GPU infrastructure (cost: $2M/year), (2) move from monthly to annual-only contracts, (3) raise prices 40% for existing customers, (4) expand from US to EU market. Timeline: all within 6 months. Current burn rate: $500K/month, runway: 14 months.",
        "expert_criteria": ["analyzes financial viability", "assesses churn risk from pricing", "evaluates EU expansion complexity", "questions timeline feasibility", "recommends sequencing", "identifies customer retention risks"]
    },
    "research_methodology": {
        "name": "Research Methodology Review",
        "task": "Evaluate the methodology of the following proposed study.",
        "test_input": "Study: 'Effect of Framework Injection on LLM Output Quality'. Design: Compare 3 conditions (PE, CE, FI) across 6 domains. N=30 queries per condition per domain (540 total). Evaluation: 3 domain experts rate each output 1-5 on 6 criteria (blind). Analysis: ANOVA with post-hoc Tukey HSD. Power analysis: not reported. Control for model temperature: fixed at 0.7. No inter-rater reliability measure planned.",
        "expert_criteria": ["identifies missing power analysis", "flags inter-rater reliability gap", "questions sample size adequacy", "evaluates blind evaluation design", "suggests additional controls", "recommends statistical improvements"]
    }
}

# ============================================================
# THREE CONDITIONS: PE, CE, FI
# ============================================================


def get_pe_prompt(domain_info: dict) -> str:
    """Prompt Engineering condition: simple, direct instruction."""
    return domain_info["task"]


def get_ce_prompt(domain_info: dict) -> str:
    """Context Engineering condition: task + rich context."""
    return f"""You are an expert in {domain_info['name']}.

Context: You have extensive experience in this field. Consider all relevant factors, best practices, and industry standards.

Task: {domain_info['task']}

Please provide a thorough, well-structured analysis.

Input:
{domain_info['test_input']}"""


def get_fi_prompt(domain_info: dict, framework: str) -> str:
    """Framework Injection condition: full injectable framework + task."""
    return f"""{framework}

---

Now apply this framework to the following:

Task: {domain_info['task']}

Input:
{domain_info['test_input']}"""


# ============================================================
# EVALUATION
# ============================================================

EVAL_PROMPT = """You are an expert evaluator assessing the quality of a professional analysis.

The analysis was produced for the domain: {domain_name}
Task: {task}

EVALUATION CRITERIA (rate each 1-5):

1. DOMAIN ACCURACY — Are the facts, concepts, and terminology correct for this domain?
2. REASONING DEPTH — Does the analysis show systematic, multi-step reasoning (not just surface-level)?
3. COMPLETENESS — Are all important aspects addressed? (Expected aspects: {criteria})
4. ACTIONABILITY — Are the recommendations specific enough to act on?
5. HALLUCINATION CHECK — Is there any fabricated information? (5=none detected, 1=significant fabrication)
6. PROFESSIONAL TONE — Would a domain expert consider this output professional-grade?

Rate the following analysis:

---
{output}
---

Respond ONLY with a JSON object:
{{"domain_accuracy": X, "reasoning_depth": X, "completeness": X, "actionability": X, "hallucination_check": X, "professional_tone": X, "total": X, "brief_justification": "..."}}

Where "total" is the sum of all 6 scores (max 30).
"""


def evaluate_output(domain_info: dict, output: str, model: str = "claude-sonnet-4-6") -> dict:
    """Evaluate a single output using LLM-as-judge."""
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": EVAL_PROMPT.format(
                domain_name=domain_info["name"],
                task=domain_info["task"],
                criteria=", ".join(domain_info["expert_criteria"]),
                output=output
            )
        }]
    )

    try:
        text = message.content[0].text
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse evaluation", "raw": text}


def run_condition(domain_info: dict, prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Run a single condition and return the output."""
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": prompt + "\n\nInput:\n" + domain_info["test_input"]
                if "Input:" not in prompt else prompt
        }]
    )

    return message.content[0].text


# ============================================================
# MAIN BENCHMARK
# ============================================================

def run_benchmark(domains: list = None, runs: int = 3, output_dir: str = "results"):
    """Run the full PE vs CE vs FI benchmark."""

    if domains is None:
        domains = list(DOMAINS.keys())

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("frameworks").mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "timestamp": timestamp,
            "domains": domains,
            "runs_per_condition": runs,
            "model": "claude-sonnet-4-6",
            "temperature": 0.7
        },
        "results": []
    }

    # Step 1: Generate frameworks for all domains
    print("=" * 60)
    print("STEP 1: Generating frameworks for FI condition")
    print("=" * 60)

    frameworks = {}
    for domain_key in domains:
        domain_info = DOMAINS[domain_key]
        print(f"  Generating framework for: {domain_info['name']}...")

        fw_path = f"frameworks/{domain_key}.md"
        if os.path.exists(fw_path):
            with open(fw_path) as f:
                frameworks[domain_key] = f.read()
            print(f"    (loaded from cache)")
        else:
            from fi_generator import generate_framework, save_framework
            fw = generate_framework(domain_info["name"])
            save_framework(fw, domain_info["name"], fw_path)
            frameworks[domain_key] = fw
            print(f"    (generated and saved)")

        time.sleep(1)  # Rate limiting

    # Step 2: Run all conditions
    print("\n" + "=" * 60)
    print("STEP 2: Running PE vs CE vs FI across all domains")
    print("=" * 60)

    total_calls = len(domains) * 3 * runs  # domains x conditions x runs
    call_count = 0

    for domain_key in domains:
        domain_info = DOMAINS[domain_key]
        print(f"\n  Domain: {domain_info['name']}")

        for run_idx in range(runs):
            print(f"    Run {run_idx + 1}/{runs}")

            for condition, prompt_fn in [
                ("PE", lambda d: get_pe_prompt(d)),
                ("CE", lambda d: get_ce_prompt(d)),
                ("FI", lambda d: get_fi_prompt(d, frameworks[domain_key]))
            ]:
                call_count += 1
                print(f"      [{call_count}/{total_calls}] {condition}...",
                      end=" ", flush=True)

                prompt = prompt_fn(domain_info)
                output = run_condition(domain_info, prompt)

                # Evaluate
                evaluation = evaluate_output(domain_info, output)

                result = {
                    "domain": domain_key,
                    "domain_name": domain_info["name"],
                    "condition": condition,
                    "run": run_idx + 1,
                    "output_length": len(output),
                    "evaluation": evaluation,
                    "output": output[:500] + "..." if len(output) > 500 else output
                }

                results["results"].append(result)

                total = evaluation.get("total", "N/A")
                print(f"Score: {total}")

                time.sleep(2)  # Rate limiting

    # Step 3: Aggregate results
    print("\n" + "=" * 60)
    print("STEP 3: Aggregating results")
    print("=" * 60)

    summary = aggregate_results(results["results"])
    results["summary"] = summary

    # Save results
    results_path = f"{output_dir}/benchmark_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print_summary(summary)

    # Save summary as markdown
    summary_path = f"{output_dir}/summary_{timestamp}.md"
    save_summary_markdown(summary, summary_path)

    print(f"\nResults saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")

    return results


def aggregate_results(results: list) -> dict:
    """Aggregate benchmark results by condition and domain."""
    scores = defaultdict(lambda: defaultdict(list))

    for r in results:
        condition = r["condition"]
        if "error" not in r.get("evaluation", {}):
            for criterion in ["domain_accuracy", "reasoning_depth", "completeness",
                              "actionability", "hallucination_check", "professional_tone"]:
                val = r["evaluation"].get(criterion)
                if val is not None:
                    scores[condition][criterion].append(val)

            total = r["evaluation"].get("total")
            if total is not None:
                scores[condition]["total"].append(total)

    summary = {}
    for condition in ["PE", "CE", "FI"]:
        summary[condition] = {}
        for criterion in scores[condition]:
            vals = scores[condition][criterion]
            if vals:
                summary[condition][criterion] = {
                    "mean": round(sum(vals) / len(vals), 2),
                    "min": min(vals),
                    "max": max(vals),
                    "n": len(vals)
                }

    return summary


def print_summary(summary: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: PE vs CE vs FI")
    print("=" * 70)

    criteria = ["domain_accuracy", "reasoning_depth", "completeness",
                "actionability", "hallucination_check", "professional_tone", "total"]

    header = f"{'Criterion':<25} {'PE':>8} {'CE':>8} {'FI':>8} {'FI-PE':>8}"
    print(header)
    print("-" * 57)

    for c in criteria:
        pe = summary.get("PE", {}).get(c, {}).get("mean", "N/A")
        ce = summary.get("CE", {}).get(c, {}).get("mean", "N/A")
        fi = summary.get("FI", {}).get(c, {}).get("mean", "N/A")

        if isinstance(pe, (int, float)) and isinstance(fi, (int, float)):
            delta = f"+{fi-pe:.2f}" if fi > pe else f"{fi-pe:.2f}"
        else:
            delta = "N/A"

        pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else str(pe)
        ce_str = f"{ce:.2f}" if isinstance(ce, (int, float)) else str(ce)
        fi_str = f"{fi:.2f}" if isinstance(fi, (int, float)) else str(fi)

        print(f"{c:<25} {pe_str:>8} {ce_str:>8} {fi_str:>8} {delta:>8}")


def save_summary_markdown(summary: dict, path: str):
    """Save summary as a markdown report."""
    with open(path, "w") as f:
        f.write("# PE vs CE vs FI Benchmark Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("| Criterion | PE | CE | FI | FI-PE |\n")
        f.write("|-----------|----|----|----|---------|\n")

        for c in ["domain_accuracy", "reasoning_depth", "completeness",
                   "actionability", "hallucination_check", "professional_tone", "total"]:
            pe = summary.get("PE", {}).get(c, {}).get("mean", "N/A")
            ce = summary.get("CE", {}).get(c, {}).get("mean", "N/A")
            fi = summary.get("FI", {}).get(c, {}).get("mean", "N/A")

            if isinstance(pe, (int, float)) and isinstance(fi, (int, float)):
                delta = f"+{fi-pe:.2f}" if fi > pe else f"{fi-pe:.2f}"
            else:
                delta = "N/A"

            f.write(f"| {c} | {pe} | {ce} | {fi} | {delta} |\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PE vs CE vs FI Benchmark")
    parser.add_argument("--domains", default="all",
                        help="Comma-separated domains or 'all'")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per condition")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    args = parser.parse_args()

    domains = None if args.domains == "all" else args.domains.split(",")
    run_benchmark(domains=domains, runs=args.runs, output_dir=args.output)
