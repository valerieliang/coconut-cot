# scripts/04_extract_reps.py
# This script runs both runners cleanly as a single reproducible step.

import sys, os
sys.path.insert(0, os.path.abspath("."))

from src.verbal_cot_runner import extract_verbal_representations
from src.coconut_runner import extract_coconut_representations

if __name__ == "__main__":
    print("=" * 50)
    print("Extracting verbal CoT representations...")
    print("=" * 50)
    extract_verbal_representations(
        data_path="data/prontoqa_split.csv",
        split="test",
        output_path="data/verbal_reps.json",
    )

    print()
    print("=" * 50)
    print("Extracting Coconut representations...")
    print("=" * 50)
    extract_coconut_representations(
        data_path="data/prontoqa_split.csv",
        split="test",
        output_path="data/coconut_reps.json",
    )

    print()
    print("Both extraction complete.")
    print("  data/verbal_reps.json")
    print("  data/coconut_reps.json")