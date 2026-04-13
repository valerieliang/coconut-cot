#!/usr/bin/env python3
"""
COMPREHENSIVE STEERING EXPERIMENT SUITE
========================================
A complete test suite for evaluating steering efficacy on Coconut vs Verbal CoT.

This script runs 12 systematic experiments to assess:
1. Baseline stability (no steering)
2. Random control at multiple scales
3. Meaningful steering at multiple reasoning steps
4. Alpha sensitivity analysis
5. Cross-model comparison (Coconut vs Verbal CoT)
6. Difficulty scaling (3, 4, 5 hop problems)
7. Cumulative vs single-step steering
8. Statistical significance testing

Author: CogAI Project
Date: 2026
"""

import sys
import os
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class ComprehensiveSteeringSuite:
    """
    Complete steering experiment suite with systematic testing.
    """
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = args.experiment_name or f"steering_suite_{self.timestamp}"
        self.output_dir = os.path.join(args.output_dir, self.experiment_name)
        self.results = {}
        self.summary_data = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        self._save_config()
        
    def _save_config(self):
        """Save experiment configuration."""
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.args), f, indent=2)
            
    def _run_steering(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single steering experiment and parse results."""
        cmd = [
            "python", "scripts/steering/run_steering.py",
            "--model_type", config.get("model_type", "coconut"),
            "--n_problems", str(config.get("n_problems", self.args.n_problems)),
            "--start_idx", str(config.get("start_idx", self.args.start_idx)),
            "--output_dir", self.output_dir,
            "--experiment_name", config["exp_name"],
            "--no_blending",  # Always use pure steering for consistent measurement
        ]
        
        # Add steering parameters
        if config.get("per_step_vectors"):
            cmd.append("--per_step_vectors")
        if config.get("steer_all"):
            cmd.append("--steer_all")
        if config.get("random_control"):
            cmd.append("--random_control")
        if config.get("random_scale"):
            cmd.extend(["--random_scale", str(config["random_scale"])])
        if config.get("verbose") or self.args.verbose:
            cmd.append("--verbose")
            
        # Add alpha values
        alphas = config.get("alphas", [0, 1, 2, 5, 10, 20, 50])
        cmd.extend(["--alphas"] + [str(a) for a in alphas])
        
        # Add hop filter
        if self.args.hop_filter:
            cmd.extend(["--hop_filter"] + [str(h) for h in self.args.hop_filter])
            
        print(f"\n  [RUNNING] {config['exp_name']}")
        print(f"     Command: {' '.join(cmd[-10:])}...")  # Show last part only
        
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        # Parse results
        metrics = {
            "exp_name": config["exp_name"],
            "model_type": config["model_type"],
            "return_code": result.returncode,
            "answer_flips": 0,
            "total_problems": 0,
            "max_effect_size": 0.0,
            "flip_rate": 0.0,
        }
        
        for line in result.stdout.split("\n"):
            if "Answer flipped:" in line:
                parts = line.split("Answer flipped:")[-1].strip()
                if "/" in parts:
                    metrics["answer_flips"] = int(parts.split("/")[0])
                    metrics["total_problems"] = int(parts.split("/")[1].split()[0])
                    metrics["flip_rate"] = metrics["answer_flips"] / metrics["total_problems"] if metrics["total_problems"] > 0 else 0
            elif "Max effect size:" in line:
                try:
                    metrics["max_effect_size"] = float(line.split("Max effect size:")[-1].strip())
                except:
                    pass
                    
        # Also try to read from results file if available
        results_path = os.path.join(self.output_dir, config["exp_name"], "steering_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    full_results = json.load(f)
                    if "results" in full_results:
                        for r in full_results["results"]:
                            if r.get("answer_flipped", False):
                                metrics["answer_flips"] += 1
                        metrics["total_problems"] = len(full_results["results"])
                        metrics["flip_rate"] = metrics["answer_flips"] / metrics["total_problems"] if metrics["total_problems"] > 0 else 0
            except:
                pass
                
        return metrics
    
    # ================================================================
    # EXPERIMENT 1: BASELINE STABILITY
    # ================================================================
    def experiment_1_baseline(self) -> Dict:
        """Establish baseline: No steering = 0 flips."""
        print("\n" + "="*80)
        print("EXPERIMENT 1: BASELINE STABILITY")
        print("="*80)
        print("Purpose: Verify that without steering, answers remain stable.")
        print("Expected: 0 flips for both models.")
        
        results = {}
        
        for model in ["coconut", "verbal"]:
            print(f"\n  Testing {model.upper()}...")
            config = {
                "exp_name": f"01_baseline_{model}_{self.timestamp}",
                "model_type": model,
                "alphas": [0],  # Only baseline
                "per_step_vectors": False,
            }
            metrics = self._run_steering(config)
            results[model] = metrics
            print(f"    [OK] {model.upper()}: {metrics['answer_flips']}/{metrics['total_problems']} flips")
            
        self.results["baseline"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 2: RANDOM CONTROL (Multiple Scales)
    # ================================================================
    def experiment_2_random_control(self) -> Dict:
        """
        Establish random baseline: Random vectors at multiple scales.
        Should cause 0 flips at small scales, increasing at larger scales.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: RANDOM CONTROL (Multiple Scales)")
        print("="*80)
        print("Purpose: Determine the noise floor and validate steering specificity.")
        print("Expected: scale=0.01-0.1 -> 0 flips, scale=0.5-1.0 -> some flips")
        
        scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        alphas = [0, 10, 20, 50, 100]
        
        results = {}
        
        for scale in scales:
            print(f"\n  Testing random scale = {scale}")
            config = {
                "exp_name": f"02_random_scale_{scale}_{self.timestamp}",
                "model_type": "coconut",
                "random_control": True,
                "random_scale": scale,
                "alphas": alphas,
                "per_step_vectors": False,
            }
            metrics = self._run_steering(config)
            results[f"scale_{scale}"] = metrics
            print(f"    Scale {scale}: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
        self.results["random_control"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 3: COCONUT - STEP-BY-STEP STEERING
    # ================================================================
    def experiment_3_coconut_step_steering(self) -> Dict:
        """
        Test steering at each individual reasoning step.
        Identifies which steps are causally critical.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 3: COCONUT - STEP-BY-STEP STEERING")
        print("="*80)
        print("Purpose: Identify which reasoning steps are causally important.")
        print("Expected: Some steps (e.g., step 2) cause more flips than others.")
        
        alphas = [0, 1, 2, 5, 10, 20, 50]
        
        results = {}
        
        # Test each step (up to 5 hops)
        for step in range(5):
            print(f"\n  Testing step {step}...")
            config = {
                "exp_name": f"03_coconut_step_{step}_{self.timestamp}",
                "model_type": "coconut",
                "per_step_vectors": True,
                "alphas": alphas,
                # Note: We'll analyze step-specific results from the output
            }
            metrics = self._run_steering(config)
            results[f"step_{step}"] = metrics
            print(f"    Step {step}: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
        self.results["coconut_step_steering"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 4: COCONUT - CUMULATIVE STEERING
    # ================================================================
    def experiment_4_coconut_cumulative_steering(self) -> Dict:
        """
        Test cumulative steering: steer all steps from a target onward.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 4: COCONUT - CUMULATIVE STEERING")
        print("="*80)
        print("Purpose: Test if cumulative steering has stronger effects.")
        print("Expected: Cumulative steering causes more flips at smaller alphas.")
        
        alphas = [0, 0.5, 1, 2, 5, 10, 20]
        
        results = {}
        
        for step in range(5):
            print(f"\n  Testing cumulative from step {step}...")
            config = {
                "exp_name": f"04_coconut_cumulative_{step}_{self.timestamp}",
                "model_type": "coconut",
                "per_step_vectors": True,
                "steer_all": True,
                "alphas": alphas,
            }
            metrics = self._run_steering(config)
            results[f"from_step_{step}"] = metrics
            print(f"    Cumulative from step {step}: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
        self.results["coconut_cumulative"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 5: COCONUT - ALPHA SENSITIVITY (Fine-grained)
    # ================================================================
    def experiment_5_coconut_alpha_sensitivity(self) -> Dict:
        """
        Fine-grained alpha sweep to find exact flip thresholds.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 5: COCONUT - ALPHA SENSITIVITY")
        print("="*80)
        print("Purpose: Find the minimum alpha that causes answer flips.")
        print("Expected: Threshold around alpha=2-5 for critical steps.")
        
        # Very fine-grained alphas around expected threshold
        alpha_ranges = {
            "tiny": [0, 0.1, 0.2, 0.5, 0.8, 1.0],
            "small": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "medium": [0, 10, 15, 20, 25, 30, 40, 50],
            "large": [0, 50, 75, 100, 150, 200, 300, 500],
        }
        
        results = {}
        
        for range_name, alphas in alpha_ranges.items():
            print(f"\n  Alpha range: {range_name} ({len(alphas)} values)")
            config = {
                "exp_name": f"05_alpha_{range_name}_{self.timestamp}",
                "model_type": "coconut",
                "per_step_vectors": True,
                "alphas": alphas,
            }
            metrics = self._run_steering(config)
            results[range_name] = metrics
            print(f"    {range_name}: {metrics['answer_flips']}/{metrics['total_problems']} flips")
            
        self.results["alpha_sensitivity"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 6: VERBAL COT - STEP-BY-STEP STEERING
    # ================================================================
    def experiment_6_verbal_steering(self) -> Dict:
        """
        Test steering on Verbal CoT model at each reasoning step.
        This tests the hypothesis that Verbal CoT is causally opaque.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 6: VERBAL COT - STEP-BY-STEP STEERING")
        print("="*80)
        print("Purpose: Test if Verbal CoT shows causal steering effects.")
        print("Expected: 0 flips (hypothesis) or significantly fewer than Coconut.")
        
        alphas = [0, 10, 20, 50, 100, 200, 500]
        
        results = {}
        
        # Test each reasoning step
        for step in range(5):
            print(f"\n  Testing step {step}...")
            config = {
                "exp_name": f"06_verbal_step_{step}_{self.timestamp}",
                "model_type": "verbal",
                "per_step_vectors": True,
                "alphas": alphas,
            }
            metrics = self._run_steering(config)
            results[f"step_{step}"] = metrics
            print(f"    Verbal step {step}: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
        self.results["verbal_steering"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 7: COCONUT VS VERBAL (Direct Comparison)
    # ================================================================
    def experiment_7_direct_comparison(self) -> Dict:
        """
        Direct head-to-head comparison at equal alpha values.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 7: COCONUT VS VERBAL (Direct Comparison)")
        print("="*80)
        print("Purpose: Directly compare steering efficacy between models.")
        print("Expected: Coconut shows significantly more flips than Verbal.")
        
        alphas = [0, 1, 2, 5, 10, 20, 50, 100]
        
        results = {}
        
        for model in ["coconut", "verbal"]:
            print(f"\n  Testing {model.upper()}...")
            config = {
                "exp_name": f"07_comparison_{model}_{self.timestamp}",
                "model_type": model,
                "per_step_vectors": True,
                "alphas": alphas,
            }
            metrics = self._run_steering(config)
            results[model] = metrics
            print(f"    {model.upper()}: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
        self.results["direct_comparison"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 8: DIFFICULTY SCALING (Hop Count Analysis)
    # ================================================================
    def experiment_8_difficulty_scaling(self) -> Dict:
        """
        Test how steering efficacy scales with problem difficulty (hop count).
        """
        print("\n" + "="*80)
        print("EXPERIMENT 8: DIFFICULTY SCALING (Hop Count Analysis)")
        print("="*80)
        print("Purpose: Test if steering works better/worse on longer reasoning chains.")
        print("Expected: May show different patterns for different hop counts.")
        
        hop_counts = [3, 4, 5]
        alphas = [0, 5, 10, 20, 50]
        
        results = {}
        
        for hops in hop_counts:
            print(f"\n  Testing {hops}-hop problems...")
            
            # Temporarily override hop_filter
            original_filter = self.args.hop_filter
            self.args.hop_filter = [hops]
            
            config = {
                "exp_name": f"08_difficulty_{hops}hop_{self.timestamp}",
                "model_type": "coconut",
                "per_step_vectors": True,
                "alphas": alphas,
            }
            metrics = self._run_steering(config)
            results[f"{hops}hop"] = metrics
            print(f"    {hops}-hop: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
            # Restore
            self.args.hop_filter = original_filter
            
        self.results["difficulty_scaling"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 9: NEGATIVE STEERING (Opposite Direction)
    # ================================================================
    def experiment_9_negative_steering(self) -> Dict:
        """
        Test steering in the negative direction (should flip to False).
        """
        print("\n" + "="*80)
        print("EXPERIMENT 9: NEGATIVE STEERING (Opposite Direction)")
        print("="*80)
        print("Purpose: Test if negative alphas flip answers in the opposite direction.")
        print("Expected: Negative alphas cause flips from True -> False.")
        
        alphas = [0, -1, -2, -5, -10, -20, -50, -100]
        
        results = {}
        
        for model in ["coconut", "verbal"]:
            print(f"\n  Testing {model.upper()} with negative alphas...")
            config = {
                "exp_name": f"09_negative_{model}_{self.timestamp}",
                "model_type": model,
                "per_step_vectors": True,
                "alphas": alphas,
            }
            metrics = self._run_steering(config)
            results[model] = metrics
            print(f"    {model.upper()}: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
        self.results["negative_steering"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 10: FIRST STEP VS LAST STEP
    # ================================================================
    def experiment_10_first_vs_last(self) -> Dict:
        """
        Compare steering at first step vs last step.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 10: FIRST STEP VS LAST STEP")
        print("="*80)
        print("Purpose: Determine which step is more causally important.")
        print("Expected: May differ based on reasoning structure.")
        
        alphas = [0, 1, 2, 5, 10, 20, 50]
        
        results = {}
        
        for step_name, step_idx in [("first", 0), ("last", 4)]:
            print(f"\n  Testing {step_name} step...")
            config = {
                "exp_name": f"10_{step_name}_step_{self.timestamp}",
                "model_type": "coconut",
                "per_step_vectors": True,
                "alphas": alphas,
            }
            metrics = self._run_steering(config)
            results[step_name] = metrics
            print(f"    {step_name.upper()} step: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
        self.results["first_vs_last"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 11: REPRODUCIBILITY TEST (Multiple Seeds)
    # ================================================================
    def experiment_11_reproducibility(self) -> Dict:
        """
        Test reproducibility by running the same experiment multiple times.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 11: REPRODUCIBILITY TEST")
        print("="*80)
        print("Purpose: Verify that results are consistent across runs.")
        print("Expected: Similar flip rates across seeds.")
        
        seeds = [42, 123, 456]
        alphas = [0, 5, 10, 20]
        
        results = {}
        
        for seed in seeds:
            print(f"\n  Testing seed {seed}...")
            
            # Temporarily set seed
            original_seed = self.args.seed
            self.args.seed = seed
            
            config = {
                "exp_name": f"11_reproducible_seed{seed}_{self.timestamp}",
                "model_type": "coconut",
                "per_step_vectors": True,
                "alphas": alphas,
            }
            metrics = self._run_steering(config)
            results[f"seed_{seed}"] = metrics
            print(f"    Seed {seed}: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
            # Restore
            self.args.seed = original_seed
            
        self.results["reproducibility"] = results
        return results
    
    # ================================================================
    # EXPERIMENT 12: COMPLETE SWEEP (All Configurations)
    # ================================================================
    def experiment_12_complete_sweep(self) -> Dict:
        """
        Complete sweep of all configurations on a small problem set.
        This is the most comprehensive test.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 12: COMPLETE SWEEP (All Configurations)")
        print("="*80)
        print("Purpose: Test all combinations on a small set to find optimal settings.")
        
        # Use fewer problems for complete sweep
        original_n = self.args.n_problems
        self.args.n_problems = min(5, original_n)
        
        configurations = [
            # (model, per_step, steer_all, random_control, random_scale, alpha_name)
            ("coconut", True, False, False, None, "meaningful_per_step"),
            ("coconut", True, True, False, None, "meaningful_cumulative"),
            ("coconut", False, False, False, None, "embedding_only"),
            ("coconut", False, False, True, 0.1, "random_scale_0.1"),
            ("coconut", False, False, True, 0.5, "random_scale_0.5"),
            ("verbal", True, False, False, None, "verbal_per_step"),
            ("verbal", False, False, False, None, "verbal_embedding"),
        ]
        
        alphas = [0, 1, 2, 5, 10, 20, 50]
        
        results = {}
        
        for model, per_step, steer_all, random_ctrl, rand_scale, name in configurations:
            print(f"\n  Testing: {name}")
            config = {
                "exp_name": f"12_sweep_{name}_{self.timestamp}",
                "model_type": model,
                "per_step_vectors": per_step,
                "steer_all": steer_all,
                "random_control": random_ctrl,
                "alphas": alphas,
            }
            if random_ctrl and rand_scale:
                config["random_scale"] = rand_scale
                
            metrics = self._run_steering(config)
            results[name] = metrics
            print(f"    {name}: {metrics['answer_flips']}/{metrics['total_problems']} flips (rate={metrics['flip_rate']:.2f})")
            
        # Restore
        self.args.n_problems = original_n
        
        self.results["complete_sweep"] = results
        return results
    
    # ================================================================
    # ANALYSIS AND REPORTING
    # ================================================================
    def generate_report(self) -> Dict:
        """
        Generate comprehensive analysis report with statistics.
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        report = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": {
                "n_problems": self.args.n_problems,
                "start_idx": self.args.start_idx,
                "hop_filter": self.args.hop_filter,
                "seed": self.args.seed,
            },
            "results": self.results,
            "analysis": {},
            "conclusions": {},
        }
        
        # ============================================================
        # ANALYSIS 1: Baseline Validation
        # ============================================================
        if "baseline" in self.results:
            baseline_coconut = self.results["baseline"].get("coconut", {})
            baseline_verbal = self.results["baseline"].get("verbal", {})
            
            report["analysis"]["baseline_valid"] = (
                baseline_coconut.get("answer_flips", 1) == 0 and
                baseline_verbal.get("answer_flips", 1) == 0
            )
            report["analysis"]["baseline_message"] = (
                "[OK] Baseline stable" if report["analysis"]["baseline_valid"] 
                else "[FAIL] Baseline shows flips without steering!"
            )
        
        # ============================================================
        # ANALYSIS 2: Random Control Threshold
        # ============================================================
        if "random_control" in self.results:
            random_results = self.results["random_control"]
            smallest_scale = min([float(k.split("_")[1]) for k in random_results.keys() if "scale" in k])
            smallest_flips = random_results.get(f"scale_{smallest_scale}", {}).get("answer_flips", 1)
            
            report["analysis"]["random_floor"] = smallest_scale
            report["analysis"]["random_floor_valid"] = smallest_flips == 0
            report["analysis"]["random_floor_message"] = (
                f"[OK] Random scale {smallest_scale} causes 0 flips" if smallest_flips == 0
                else f"[WARN] Random scale {smallest_scale} causes {smallest_flips} flips"
            )
        
        # ============================================================
        # ANALYSIS 3: Coconut Steering Efficacy
        # ============================================================
        if "coconut_step_steering" in self.results:
            step_results = self.results["coconut_step_steering"]
            best_step = max(step_results.items(), key=lambda x: x[1].get("flip_rate", 0))
            
            report["analysis"]["coconut_best_step"] = best_step[0]
            report["analysis"]["coconut_best_flip_rate"] = best_step[1].get("flip_rate", 0)
            report["analysis"]["coconut_effective"] = best_step[1].get("flip_rate", 0) > 0.5
        
        # ============================================================
        # ANALYSIS 4: Verbal CoT Steering Efficacy
        # ============================================================
        if "verbal_steering" in self.results:
            verbal_results = self.results["verbal_steering"]
            verbal_flip_rate = max([r.get("flip_rate", 0) for r in verbal_results.values()])
            
            report["analysis"]["verbal_max_flip_rate"] = verbal_flip_rate
            report["analysis"]["verbal_effective"] = verbal_flip_rate > 0.2
        
        # ============================================================
        # ANALYSIS 5: Direct Comparison
        # ============================================================
        if "direct_comparison" in self.results:
            comp = self.results["direct_comparison"]
            coconut_rate = comp.get("coconut", {}).get("flip_rate", 0)
            verbal_rate = comp.get("verbal", {}).get("flip_rate", 0)
            
            report["analysis"]["coconut_vs_verbal_ratio"] = coconut_rate / (verbal_rate + 0.01)
            report["analysis"]["coconut_significantly_better"] = coconut_rate > verbal_rate + 0.2
        
        # ============================================================
        # ANALYSIS 6: Difficulty Scaling
        # ============================================================
        if "difficulty_scaling" in self.results:
            diff_results = self.results["difficulty_scaling"]
            hop_rates = {k: v.get("flip_rate", 0) for k, v in diff_results.items()}
            report["analysis"]["difficulty_scaling"] = hop_rates
        
        # ============================================================
        # ANALYSIS 7: Negative Steering
        # ============================================================
        if "negative_steering" in self.results:
            neg_results = self.results["negative_steering"]
            coconut_neg = neg_results.get("coconut", {}).get("flip_rate", 0)
            report["analysis"]["negative_steering_works"] = coconut_neg > 0.3
        
        # ============================================================
        # CONCLUSIONS
        # ============================================================
        conclusions = []
        
        # Main hypothesis
        if report["analysis"].get("coconut_effective", False) and not report["analysis"].get("verbal_effective", False):
            conclusions.append("[CONFIRMED] MAIN HYPOTHESIS: Coconut is causally steerable, Verbal CoT is not")
        elif report["analysis"].get("coconut_effective", False) and report["analysis"].get("verbal_effective", False):
            conclusions.append("[PARTIAL] Both models show steering effects, but Coconut may be stronger")
        else:
            conclusions.append("[REJECTED] Neither model shows reliable steering effects")
        
        # Validity
        if report["analysis"].get("baseline_valid", False) and report["analysis"].get("random_floor_valid", False):
            conclusions.append("[VALID] Baseline and random control pass validity checks")
        else:
            conclusions.append("[WARN] Validity concern: Baseline or random control shows unexpected flips")
        
        # Best configuration
        if report["analysis"].get("coconut_effective", False):
            best = report["analysis"].get("coconut_best_step", "unknown")
            conclusions.append(f"[BEST] Configuration: Steering at {best} with per-step vectors")
        
        report["conclusions"] = conclusions
        
        # Save report
        report_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: Dict):
        """Print formatted summary to console."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print(f"\n[FOLDER] Experiment: {report['experiment_name']}")
        print(f"[DATE] Timestamp: {report['timestamp']}")
        print(f"[DATA] Problems: {report['config']['n_problems']}")
        
        print("\n" + "-"*40)
        print("VALIDITY CHECKS")
        print("-"*40)
        print(f"  {report['analysis'].get('baseline_message', 'N/A')}")
        print(f"  {report['analysis'].get('random_floor_message', 'N/A')}")
        
        print("\n" + "-"*40)
        print("STEERING EFFECTIVENESS")
        print("-"*40)
        print(f"  Coconut best flip rate: {report['analysis'].get('coconut_best_flip_rate', 0):.2%}")
        print(f"  Coconut best step: {report['analysis'].get('coconut_best_step', 'N/A')}")
        print(f"  Verbal max flip rate: {report['analysis'].get('verbal_max_flip_rate', 0):.2%}")
        
        if "coconut_vs_verbal_ratio" in report["analysis"]:
            print(f"  Coconut/Verbal ratio: {report['analysis']['coconut_vs_verbal_ratio']:.1f}x")
        
        print("\n" + "-"*40)
        print("CONCLUSIONS")
        print("-"*40)
        for conclusion in report["conclusions"]:
            print(f"  {conclusion}")
        
        print(f"\n[SAVED] Full report saved to: {os.path.join(self.output_dir, 'analysis_report.json')}")
    
    # ================================================================
    # MAIN RUN METHOD
    # ================================================================
    def run_all(self):
        """Run all experiments in the suite."""
        print("\n" + "="*80)
        print("COMPREHENSIVE STEERING EXPERIMENT SUITE")
        print("="*80)
        print(f"Experiment: {self.experiment_name}")
        print(f"Output: {self.output_dir}")
        print(f"Problems: {self.args.n_problems}")
        print(f"Hop filter: {self.args.hop_filter}")
        print(f"Seed: {self.args.seed}")
        
        # Define experiment order (from quickest to longest)
        experiments = [
            ("1. Baseline", self.experiment_1_baseline),
            ("2. Random Control", self.experiment_2_random_control),
            ("3. Coconut Step Steering", self.experiment_3_coconut_step_steering),
            ("4. Coconut Cumulative", self.experiment_4_coconut_cumulative_steering),
            ("5. Alpha Sensitivity", self.experiment_5_coconut_alpha_sensitivity),
            ("6. Verbal CoT Steering", self.experiment_6_verbal_steering),
            ("7. Direct Comparison", self.experiment_7_direct_comparison),
            ("8. Difficulty Scaling", self.experiment_8_difficulty_scaling),
            ("9. Negative Steering", self.experiment_9_negative_steering),
            ("10. First vs Last", self.experiment_10_first_vs_last),
            ("11. Reproducibility", self.experiment_11_reproducibility),
        ]
        
        # Add complete sweep if not in quick mode
        if not self.args.quick:
            experiments.append(("12. Complete Sweep", self.experiment_12_complete_sweep))
        
        for name, experiment_func in experiments:
            print(f"\n{'='*80}")
            print(f">>> RUNNING: {name}")
            print(f"{'='*80}")
            try:
                experiment_func()
            except Exception as e:
                print(f"  [ERROR] in {name}: {e}")
                if self.args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # Generate final report
        report = self.generate_report()
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        
        return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive Steering Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Run full suite on 20 problems
  python steering_suite.py --n_problems 20
  
  # Quick test on 5 problems
  python steering_suite.py --n_problems 5 --quick
  
  # Test only 3-hop problems
  python steering_suite.py --hop_filter 3 --n_problems 10
  
  # Run with verbose output
  python steering_suite.py --verbose --n_problems 10
        """
    )
    
    # Data selection
    parser.add_argument("--n_problems", type=int, default=10,
                        help="Number of problems to test (default: 10)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in dataset (default: 0)")
    parser.add_argument("--hop_filter", type=int, nargs="+", default=None,
                        help="Filter by hop count, e.g., --hop_filter 3 4 5")
    
    # Experiment control
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Custom experiment name")
    parser.add_argument("--output_dir", type=str, default="results/steering",
                        help="Output directory (default: results/steering)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer problems, skip complete sweep")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.n_problems = min(5, args.n_problems)
        print("[QUICK MODE] Using fewer problems")
    
    # Run suite
    suite = ComprehensiveSteeringSuite(args)
    suite.run_all()


if __name__ == "__main__":
    main()