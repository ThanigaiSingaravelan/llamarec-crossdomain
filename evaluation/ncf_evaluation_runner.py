#!/usr/bin/env python3
"""
NCF Evaluation Runner - Fixed version with proper imports
"""

import argparse
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.neural_cf import evaluate_ncf_with_ground_truth  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate NCF baseline on a dataset")
    parser.add_argument("--dataset", required=True, help="Path to CSV file (e.g., Books_filtered.csv)")
    parser.add_argument("--user_history", required=True, help="Path to user_history.json")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., Books)")
    parser.add_argument("--output", default="results/ncf_results.json", help="Output path for results JSON")
    parser.add_argument("--k", type=int, default=3, help="Top-K for evaluation")

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        print("💡 Make sure you've run the data preprocessing pipeline first:")
        print("   python preprocessing/domain_preprocessor.py")
        return

    if not os.path.exists(args.user_history):
        print(f"❌ User history file not found: {args.user_history}")
        print("💡 Make sure you've run the data preprocessing pipeline first:")
        print("   python preprocessing/domain_preprocessor.py")
        return

    print(f"🚀 Running NCF Evaluation")
    print(f"Dataset: {args.dataset}")
    print(f"Domain: {args.domain}")
    print(f"K: {args.k}")
    print("=" * 50)

    # Load and validate dataset
    try:
        df = pd.read_csv(args.dataset)
        print(f"✅ Loaded dataset: {len(df)} interactions")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Run NCF evaluation
    try:
        evaluate_ncf_with_ground_truth(
            df,
            user_history_path=args.user_history,
            domain=args.domain,
            output_path=args.output,
            k=args.k
        )
        print(f"\n🎉 NCF evaluation completed successfully!")
        print(f"📊 Results saved to: {args.output}")

    except Exception as e:
        print(f"❌ NCF evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()