#!/usr/bin/env python3
"""
Framework Injection Generator v2.0 — Adaptive by Model Capability

Key insight from benchmark data: frontier models perform WORSE with
full instructional frameworks (scaffolding becomes ceiling). They need
META-COGNITIVE frameworks instead.

v2 generates different framework types based on target model:
- "scaffold" mode (for Haiku/smaller): all 5 types, explicit, detailed
- "metacognitive" mode (for Sonnet/Opus/frontier): evaluative + ethical +
  meta-cognitive monitoring. No procedural instructions that override
  the model's native capabilities.

Usage:
    python3 fi_generator_v2.py --domain "corporate law" --mode scaffold
    python3 fi_generator_v2.py --domain "corporate law" --mode metacognitive
    python3 fi_generator_v2.py --domain "corporate law" --mode auto --target-model sonnet
"""

import anthropic
import argparse
from pathlib import Path

# ============================================================
# META-FRAMEWORK: SCAFFOLD MODE (for smaller models)
# ============================================================

SCAFFOLD_META = """You are a Framework Injection specialist generating a SCAFFOLD framework
for a SMALLER language model that needs explicit guidance.

Generate a COMPLETE framework with ALL FIVE types — be detailed and explicit:

1. DECLARATIVE — Full knowledge hierarchy, source rankings, key concepts defined
2. PROCEDURAL — Detailed step-by-step reasoning method with decision points
3. EVALUATIVE — Explicit quality criteria and scoring rubric
4. ETHICAL — All hard constraints, compliance requirements, absolute limits
5. COMPOSITIONAL — Complete workflow integrating all above

Use HIGH semiotic density terms (domain-specific, not generic).
Be thorough — the model needs this scaffolding to perform above baseline.

DOMAIN: {domain}
"""

# ============================================================
# META-FRAMEWORK: METACOGNITIVE MODE (for frontier models)
# ============================================================

METACOGNITIVE_META = """You are a Framework Injection specialist generating a META-COGNITIVE framework
for a FRONTIER language model that already possesses strong domain reasoning.

CRITICAL: Do NOT tell the model HOW to reason about {domain}. It already knows.
Instead, tell the model HOW TO MONITOR AND AUDIT its own reasoning.

Generate a framework with THREE components:

1. EVALUATIVE STANDARDS (30% of framework)
   Define the quality bar. Not "how to analyze" but "what makes an analysis excellent
   vs merely adequate in {domain}."
   - What separates expert-level from competent-level output?
   - What specific dimensions matter most in this domain?
   - What does a senior practitioner look for when reviewing work?

2. ETHICAL FENCES (30% of framework)
   Define absolute constraints. Hard lines that must never be crossed.
   - What must the model NEVER claim without evidence?
   - What regulatory/compliance boundaries exist?
   - What constitutes malpractice or professional misconduct in this domain?
   - When should the model explicitly say "I don't know" or "this requires human judgment"?

3. META-COGNITIVE CHECKPOINTS (40% of framework)
   These are self-audit questions the model must apply BEFORE delivering output.
   Generate 8-12 checkpoints in this format:

   Before concluding, verify:
   □ Have I cited specific evidence for every factual claim?
   □ Have I considered the strongest counter-argument?
   □ Have I explicitly stated what I DON'T know or can't verify?
   □ Would my analysis survive adversarial review by a domain expert?
   □ Am I using domain terminology with precision, or generalizing?
   □ Have I distinguished between established consensus and my inference?
   □ [domain-specific checkpoints...]

   Each checkpoint should catch a SPECIFIC failure mode common in {domain} analysis.

SEMIOTIC DENSITY: Use the most precise domain terms available. Each term should
activate a specific professional frame. Do NOT explain basics — the model already
knows them. Reference domain concepts by their proper names.

DO NOT include procedural steps, knowledge hierarchies, or how-to instructions.
The model doesn't need a map — it needs a compass and guardrails.

DOMAIN: {domain}
"""

# ============================================================
# GENERATOR
# ============================================================

def generate_framework(domain: str, mode: str = "auto", target_model: str = "sonnet") -> str:
    """Generate a framework adapted to model capability."""
    client = anthropic.Anthropic()

    if mode == "auto":
        mode = "metacognitive" if target_model in ["sonnet", "opus"] else "scaffold"

    meta = METACOGNITIVE_META if mode == "metacognitive" else SCAFFOLD_META

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        messages=[{"role": "user", "content": meta.format(domain=domain)}]
    )

    return message.content[0].text


def save_framework(framework: str, domain: str, mode: str, output_path: str = None):
    """Save the generated framework."""
    if output_path is None:
        safe_name = domain.lower().replace(" ", "-").replace("/", "-")[:50]
        output_path = f"frameworks/v2/{safe_name}-{mode}.md"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# Framework Injection v2: {domain}\n")
        f.write(f"# Mode: {mode.upper()}\n\n")
        f.write(f"Generated by FI Generator v2.0 (Adaptive)\n")
        f.write(f"Methodology: Digital Craftsmanship (Gomes, 2026)\n\n")
        f.write("---\n\n")
        f.write(framework)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="FI Generator v2 — Adaptive")
    parser.add_argument("--domain", required=True, help="Domain description")
    parser.add_argument("--mode", default="auto", choices=["scaffold", "metacognitive", "auto"])
    parser.add_argument("--target-model", default="sonnet", choices=["haiku", "sonnet", "opus"])
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "metacognitive" if args.target_model in ["sonnet", "opus"] else "scaffold"

    print(f"Domain: {args.domain}")
    print(f"Mode: {mode} (target: {args.target_model})")

    framework = generate_framework(args.domain, mode, args.target_model)
    path = save_framework(framework, args.domain, mode, args.output)
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
