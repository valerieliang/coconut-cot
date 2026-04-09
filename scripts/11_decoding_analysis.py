"""
Decoding Analysis: Show Multiple Candidates in Top-k

This experiment tests three hypotheses:
1. Continuous thoughts encode multiple plausible candidates simultaneously
2. Verbal CoT representations commit to a single candidate
3. Adjacent continuous thoughts are highly similar (maintaining superposition)
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.model_utils import load_coconut_model, load_verbal_cot_model
from src.data_utils import load_split
from src.coconut_decoder import CoconutDecoder
from src.verbal_cot_decoder import VerbalCoTDecoder
from src.decoding_analyzer import DecodingAnalyzer


def run_decoding_experiment(
    test_problems=None,
    max_problems=30,
    k=10,
    device="cuda",
):
    """
    Main experiment: Decode continuous thoughts and compare to verbal CoT.
    """
    print("="*80)
    print("DECODING ANALYSIS: MULTIPLE CANDIDATES IN TOP-K")
    print("="*80)
    
    # Load data
    df_test = load_split("data/prontoqa_split.csv", split="test")
    if test_problems is None:
        # Sample problems with different hop counts
        test_problems = []
        for n_hops in [3, 4, 5]:
            subset = df_test[df_test["n_hops"] == n_hops]
            sample = subset.sample(min(len(subset), max_problems // 3), random_state=42)
            test_problems.extend([row for _, row in sample.iterrows()])
        
        if len(test_problems) > max_problems:
            test_problems = test_problems[:max_problems]
    
    print(f"Analyzing {len(test_problems)} test problems")
    
    # Load models
    print("\nLoading models...")
    coconut_model, coconut_tokenizer = load_coconut_model(device)
    verbal_model, verbal_tokenizer = load_verbal_cot_model(device)
    
    # Initialize decoders
    coconut_decoder = CoconutDecoder(coconut_model, coconut_tokenizer, device)
    verbal_decoder = VerbalCoTDecoder(verbal_model, verbal_tokenizer, device)
    
    # Run analysis
    coconut_results = []
    verbal_results = []
    
    for row in tqdm(test_problems, desc="Analyzing problems"):
        # Coconut analysis
        try:
            coco_res = coconut_decoder.analyze_thoughts(row, k=k)
            coconut_results.append(coco_res)
        except Exception as e:
            print(f"\nError on Coconut problem {row.name}: {e}")
        
        # Verbal analysis
        try:
            verb_res = verbal_decoder.analyze_steps(row, k=k)
            verbal_results.append(verb_res)
        except Exception as e:
            print(f"\nError on Verbal problem {row.name}: {e}")
    
    # Analyze results
    analyzer = DecodingAnalyzer()
    analyzer.add_results(coconut_results, verbal_results)
    summary = analyzer.generate_summary()
    
    # Print results
    print_results(summary)
    
    # Save results
    os.makedirs("results/decoding", exist_ok=True)
    
    output = {
        'summary': summary,
        'coconut_results': coconut_results,
        'verbal_results': verbal_results,
    }
    
    with open("results/decoding/full_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Results saved to results/decoding/full_results.json")
    print("="*80)
    
    return output


def print_results(summary):
    """Print formatted results."""
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)
    
    # 1. Ground truth in top-k rate
    gt_stats = summary['ground_truth_in_topk']
    print("\n1. Ground Truth in Top-10 Rate:")
    if gt_stats['coconut']['cosine'][0]:
        print(f"   Coconut (cosine): {gt_stats['coconut']['cosine'][0]:.3f} "
              f"({gt_stats['coconut']['cosine'][1]}/{gt_stats['coconut']['cosine'][2]})")
        print(f"   Coconut (logit):  {gt_stats['coconut']['logit'][0]:.3f} "
              f"({gt_stats['coconut']['logit'][1]}/{gt_stats['coconut']['logit'][2]})")
    if gt_stats['verbal']['cosine'][0]:
        print(f"   Verbal (cosine):  {gt_stats['verbal']['cosine'][0]:.3f} "
              f"({gt_stats['verbal']['cosine'][1]}/{gt_stats['verbal']['cosine'][2]})")
        print(f"   Verbal (logit):   {gt_stats['verbal']['logit'][0]:.3f} "
              f"({gt_stats['verbal']['logit'][1]}/{gt_stats['verbal']['logit'][2]})")
    
    # 2. Entropy analysis
    entropy_stats = summary['entropy']
    print("\n2. Candidate Distribution Entropy (higher = more uniform/spread out):")
    if entropy_stats['coconut']['logit']['mean']:
        print(f"   Coconut (cosine): {entropy_stats['coconut']['cosine']['mean']:.3f} +- "
              f"{entropy_stats['coconut']['cosine']['std']:.3f}")
        print(f"   Coconut (logit):  {entropy_stats['coconut']['logit']['mean']:.3f} +- "
              f"{entropy_stats['coconut']['logit']['std']:.3f}")
    if entropy_stats['verbal']['logit']['mean']:
        print(f"   Verbal (cosine):  {entropy_stats['verbal']['cosine']['mean']:.3f} +- "
              f"{entropy_stats['verbal']['cosine']['std']:.3f}")
        print(f"   Verbal (logit):   {entropy_stats['verbal']['logit']['mean']:.3f} +- "
              f"{entropy_stats['verbal']['logit']['std']:.3f}")
    
    if summary.get('entropy_ttest'):
        ttest = summary['entropy_ttest']
        print(f"   Coconut vs Verbal (logit entropy): t={ttest['t_stat']:.3f}, p={ttest['p_val']:.4f}")
    
    # 3. Adjacent thought similarity
    adj_stats = summary['adjacent_similarity']
    print("\n3. Adjacent Step Cosine Similarity:")
    if adj_stats['coconut']['mean']:
        print(f"   Coconut: {adj_stats['coconut']['mean']:.3f} +- {adj_stats['coconut']['std']:.3f} "
              f"(n={adj_stats['coconut']['n']})")
    if adj_stats['verbal']['mean']:
        print(f"   Verbal:  {adj_stats['verbal']['mean']:.3f} +- {adj_stats['verbal']['std']:.3f} "
              f"(n={adj_stats['verbal']['n']})")
    
    if summary.get('adjacent_similarity_ttest'):
        ttest = summary['adjacent_similarity_ttest']
        print(f"   Difference: t={ttest['t_stat']:.3f}, p={ttest['p_val']:.6f}")
    
    # 4. Number of unique candidates per step
    unique_stats = summary['unique_candidates']
    print("\n4. Unique Candidates in Top-10:")
    if unique_stats['coconut']['mean']:
        print(f"   Coconut: {unique_stats['coconut']['mean']:.2f} +- {unique_stats['coconut']['std']:.2f}")
    if unique_stats['verbal']['mean']:
        print(f"   Verbal:  {unique_stats['verbal']['mean']:.2f} +- {unique_stats['verbal']['std']:.2f}")
    
    # 5. Candidate overlap
    overlap_stats = summary['candidate_overlap']
    print("\n5. Jaccard Similarity of Top-5 Candidates (Consecutive Steps):")
    if overlap_stats['coconut']['mean']:
        print(f"   Coconut: {overlap_stats['coconut']['mean']:.3f} +- {overlap_stats['coconut']['std']:.3f}")
    if overlap_stats['verbal']['mean']:
        print(f"   Verbal:  {overlap_stats['verbal']['mean']:.3f} +- {overlap_stats['verbal']['std']:.3f}")
    
    print(f"\nSummary statistics:")
    print(f"   Number of problems analyzed: {summary['n_problems']}")
    print(f"   Number of Coconut steps: {summary['n_steps_coconut']}")
    print(f"   Number of Verbal steps: {summary['n_steps_verbal']}")


if __name__ == "__main__":
    run_decoding_experiment(max_problems=30, k=10)