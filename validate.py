"""Quick validation: 3 domains, 1 run each. ~9 API calls."""
from benchmark import run_benchmark

if __name__ == "__main__":
    results = run_benchmark(
        domains=["legal_contract", "medical_diagnosis", "financial_analysis"],
        runs=1,
        output_dir="results/quick"
    )
