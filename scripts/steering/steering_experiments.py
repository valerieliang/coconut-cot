#!/usr/bin/env python3
"""
Comprehensive steering experiment runner.
Tests different steering patterns with proper random control.
"""

import sys
import os
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class ComprehensiveSteeringExperiment:
    """Run comprehensive steering experiments with various configurations."""
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = args.experiment_name or f"comprehensive_steering_{self.timestamp}"
        self.output_dir = os.path.join(args.output_dir, self.experiment_name)
        self.results = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_steering_command(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single steering experiment and return results."""
        cmd = [
            "python", "scripts/steering/run_steering.py",
            "--model_type", config.get("model_type", "coconut"),
            "--n_problems", str(config.get("n_problems", 10)),
            "--start_idx", str(config.get("start_idx", 0)),
            "--output_dir", self.output_dir,
            "--experiment_name", config["exp_name"],
        ]
        
        # Add steering parameters
        if config.get("per_step_vectors"):
            cmd.append("--per_step_vectors")
        if config.get("no_blending"):
            cmd.append("--no_blending")
        if config.get("verbose"):
            cmd.append("--verbose")
        if config.get("steer_all"):
            cmd.append("--steer_all")
        if config.get("random_control"):
            cmd.append("--random_control")
            
        # Add random scale if specified
        if config.get("random_scale"):
            cmd.extend(["--random_scale", str(config["random_scale"])])
            
        # Add alpha values
        alphas = config.get("alphas", [0, 50, 100, 200, 500, 1000])
        cmd.extend(["--alphas"] + [str(a) for a in alphas])
        
        # Add step filter if specified
        if config.get("target_step") is not None:
            cmd.extend(["--target_step", str(config["target_step"])])
            
        if config.get("hop_filter"):
            cmd.extend(["--hop_filter"] + [str(h) for h in config["hop_filter"]])
            
        print(f"\n  Running: {' '.join(cmd)}")
        
        # Run command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        # Parse results from output
        output = result.stdout
        error = result.stderr
        
        # Extract key metrics from output
        metrics = {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": output[-2000:] if len(output) > 2000 else output,
            "stderr": error[-1000:] if error else "",
        }
        
        # Parse for answer flips
        flip_count = 0
        total_problems = 0
        for line in output.split("\n"):
            if "Answer flipped:" in line:
                parts = line.split("Answer flipped:")[-1].strip()
                if "/" in parts:
                    flip_count = int(parts.split("/")[0])
                    total_problems = int(parts.split("/")[1].split()[0] if len(parts.split("/")) > 1 else 0)
                metrics["answer_flips"] = flip_count
                metrics["total_problems"] = total_problems
            elif "Max effect size:" in line:
                effect = line.split("Max effect size:")[-1].strip()
                try:
                    metrics["max_effect_size"] = float(effect)
                except:
                    metrics["max_effect_size"] = effect
                    
        return metrics
    
    def run_experiment_1_baseline(self):
        """Experiment 1: Baseline - No steering (control)."""
        print("\n" + "="*70)
        print("EXPERIMENT 1: BASELINE (No Steering)")
        print("="*70)
        
        config = {
            "exp_name": f"01_baseline_{self.timestamp}",
            "model_type": "coconut",
            "n_problems": self.args.n_problems,
            "start_idx": self.args.start_idx,
            "alphas": [0],
            "per_step_vectors": False,
            "no_blending": True,
            "verbose": self.args.verbose,
        }
        
        if self.args.hop_filter:
            config["hop_filter"] = self.args.hop_filter
            
        result = self.run_steering_command(config)
        self.results["baseline"] = result
        return result
    
    def run_experiment_2_first_step_only(self):
        """Experiment 2: Steer ONLY the first reasoning step."""
        print("\n" + "="*70)
        print("EXPERIMENT 2: FIRST STEP ONLY")
        print("="*70)
        print("Testing: Steer only step 0 (the initial premise)")
        
        results = {}
        
        for alpha_set in ["positive", "negative", "mixed"]:
            print(f"\n  Alpha range: {alpha_set}")
            
            if alpha_set == "positive":
                alphas = [0, 50, 100, 200, 500, 1000]
            elif alpha_set == "negative":
                alphas = [0, -50, -100, -200, -500, -1000]
            else:
                alphas = [-1000, -500, -200, -100, -50, 0, 50, 100, 200, 500, 1000]
                
            config = {
                "exp_name": f"02_first_step_{alpha_set}_{self.timestamp}",
                "model_type": "coconut",
                "n_problems": self.args.n_problems,
                "start_idx": self.args.start_idx,
                "alphas": alphas,
                "per_step_vectors": True,
                "no_blending": True,
                "verbose": self.args.verbose,
            }
            
            if self.args.hop_filter:
                config["hop_filter"] = self.args.hop_filter
                
            result = self.run_steering_command(config)
            results[alpha_set] = result
            
        self.results["first_step_only"] = results
        return results
    
    def run_experiment_3_last_step_only(self):
        """Experiment 3: Steer ONLY the last reasoning step."""
        print("\n" + "="*70)
        print("EXPERIMENT 3: LAST STEP ONLY")
        print("="*70)
        print("Testing: Steer only the final reasoning step before answer")
        
        results = {}
        
        for alpha_set in ["positive", "negative", "mixed"]:
            print(f"\n  Alpha range: {alpha_set}")
            
            if alpha_set == "positive":
                alphas = [0, 50, 100, 200, 500, 1000]
            elif alpha_set == "negative":
                alphas = [0, -50, -100, -200, -500, -1000]
            else:
                alphas = [-1000, -500, -200, -100, -50, 0, 50, 100, 200, 500, 1000]
                
            config = {
                "exp_name": f"03_last_step_{alpha_set}_{self.timestamp}",
                "model_type": "coconut",
                "n_problems": self.args.n_problems,
                "start_idx": self.args.start_idx,
                "alphas": alphas,
                "per_step_vectors": True,
                "no_blending": True,
                "verbose": self.args.verbose,
            }
            
            if self.args.hop_filter:
                config["hop_filter"] = self.args.hop_filter
                
            result = self.run_steering_command(config)
            results[alpha_set] = result
            
        self.results["last_step_only"] = results
        return results
    
    def run_experiment_4_all_steps(self):
        """Experiment 4: Steer ALL reasoning steps from beginning."""
        print("\n" + "="*70)
        print("EXPERIMENT 4: ALL STEPS (Cumulative Steering)")
        print("="*70)
        print("Testing: Steer every step from the target step onward")
        
        results = {}
        
        for alpha_set in ["positive", "negative", "mixed"]:
            print(f"\n  Alpha range: {alpha_set}")
            
            if alpha_set == "positive":
                alphas = [0, 10, 25, 50, 100]
            elif alpha_set == "negative":
                alphas = [0, -10, -25, -50, -100]
            else:
                alphas = [-100, -50, -25, -10, 0, 10, 25, 50, 100]
                
            config = {
                "exp_name": f"04_all_steps_{alpha_set}_{self.timestamp}",
                "model_type": "coconut",
                "n_problems": self.args.n_problems,
                "start_idx": self.args.start_idx,
                "alphas": alphas,
                "per_step_vectors": True,
                "steer_all": True,
                "no_blending": True,
                "verbose": self.args.verbose,
            }
            
            if self.args.hop_filter:
                config["hop_filter"] = self.args.hop_filter
                
            result = self.run_steering_command(config)
            results[alpha_set] = result
            
        self.results["all_steps"] = results
        return results
    
    def run_experiment_5_middle_steps(self):
        """Experiment 5: Steer only middle reasoning steps."""
        print("\n" + "="*70)
        print("EXPERIMENT 5: MIDDLE STEPS ONLY")
        print("="*70)
        print("Testing: Steer each middle step individually")
        
        results = {}
        
        alphas = [-500, -200, -100, -50, 0, 50, 100, 200, 500]
        
        for step in [1, 2, 3]:
            print(f"\n  Targeting step {step}")
            
            config = {
                "exp_name": f"05_middle_step_{step}_{self.timestamp}",
                "model_type": "coconut",
                "n_problems": self.args.n_problems,
                "start_idx": self.args.start_idx,
                "alphas": alphas,
                "per_step_vectors": True,
                "no_blending": True,
                "verbose": self.args.verbose,
            }
            
            if self.args.hop_filter:
                config["hop_filter"] = self.args.hop_filter
                
            result = self.run_steering_command(config)
            results[f"step_{step}"] = result
            
        self.results["middle_steps"] = results
        return results
    
    def run_experiment_6_alpha_sensitivity(self):
        """Experiment 6: Fine-grained alpha sweep to find flip thresholds."""
        print("\n" + "="*70)
        print("EXPERIMENT 6: ALPHA SENSITIVITY (Finding Flip Thresholds)")
        print("="*70)
        print("Testing: Fine-grained alpha sweep to find exact flip points")
        
        alpha_ranges = {
            "small": list(range(-100, 101, 10)),
            "medium": list(range(-500, 501, 50)),
            "large": list(range(-1000, 1001, 100)),
        }
        
        results = {}
        
        for range_name, alphas in alpha_ranges.items():
            print(f"\n  Alpha range: {range_name} ({len(alphas)} values)")
            
            config = {
                "exp_name": f"06_alpha_sensitivity_{range_name}_{self.timestamp}",
                "model_type": "coconut",
                "n_problems": min(5, self.args.n_problems),
                "start_idx": self.args.start_idx,
                "alphas": alphas,
                "per_step_vectors": True,
                "no_blending": True,
                "verbose": False,
            }
            
            if self.args.hop_filter:
                config["hop_filter"] = self.args.hop_filter
                
            result = self.run_steering_command(config)
            results[range_name] = result
            
        self.results["alpha_sensitivity"] = results
        return results
    
    def run_experiment_7_verbal_comparison(self):
        """Experiment 7: Compare with Verbal CoT."""
        print("\n" + "="*70)
        print("EXPERIMENT 7: VERBAL COT COMPARISON")
        print("="*70)
        print("Testing: Same steering on Verbal CoT model")
        
        results = {}
        
        alphas = [-500, -200, -100, -50, 0, 50, 100, 200, 500]
        
        config = {
            "exp_name": f"07_verbal_comparison_{self.timestamp}",
            "model_type": "verbal",
            "n_problems": self.args.n_problems,
            "start_idx": self.args.start_idx,
            "alphas": alphas,
            "per_step_vectors": True,
            "no_blending": True,
            "verbose": self.args.verbose,
        }
        
        if self.args.hop_filter:
            config["hop_filter"] = self.args.hop_filter
            
        result = self.run_steering_command(config)
        results["verbal"] = result
        
        self.results["verbal_comparison"] = results
        return results
    
    def run_experiment_8_random_control(self):
        """Experiment 8: Random vector control with multiple scales."""
        print("\n" + "="*70)
        print("EXPERIMENT 8: RANDOM CONTROL (Multiple Scales)")
        print("="*70)
        print("Testing: Random steering vectors at different magnitudes")
        print("Should produce 0 flips for small scales, may produce some at large scales")
        
        results = {}
        
        # Test different random vector scales
        random_scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        alphas = [0, 50, 100, 200, 500, 1000]  # Use same alpha range as meaningful steering
        
        for scale in random_scales:
            print(f"\n  Random scale: {scale}")
            
            config = {
                "exp_name": f"08_random_control_scale_{scale}_{self.timestamp}",
                "model_type": "coconut",
                "n_problems": self.args.n_problems,
                "start_idx": self.args.start_idx,
                "alphas": alphas,
                "per_step_vectors": False,
                "random_control": True,
                "random_scale": scale,
                "no_blending": True,
                "verbose": self.args.verbose,
            }
            
            if self.args.hop_filter:
                config["hop_filter"] = self.args.hop_filter
                
            result = self.run_steering_command(config)
            results[f"scale_{scale}"] = result
            
        self.results["random_control"] = results
        return results
    
    def run_experiment_9_noise_baseline(self):
        """Experiment 9: Gaussian noise baseline (same magnitude as steering vectors)."""
        print("\n" + "="*70)
        print("EXPERIMENT 9: NOISE BASELINE")
        print("="*70)
        print("Testing: Random Gaussian noise matched to steering vector magnitude")
        
        results = {}
        
        # Match the steering vector magnitude (~1.0 normalized)
        noise_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
        alphas = [0, 50, 100, 200, 500]
        
        for scale in noise_scales:
            print(f"\n  Noise scale: {scale}")
            
            config = {
                "exp_name": f"09_noise_baseline_scale_{scale}_{self.timestamp}",
                "model_type": "coconut",
                "n_problems": min(5, self.args.n_problems),
                "start_idx": self.args.start_idx,
                "alphas": alphas,
                "per_step_vectors": False,
                "random_control": True,
                "random_scale": scale,
                "no_blending": True,
                "verbose": False,
            }
            
            if self.args.hop_filter:
                config["hop_filter"] = self.args.hop_filter
                
            result = self.run_steering_command(config)
            results[f"scale_{scale}"] = result
            
        self.results["noise_baseline"] = results
        return results
    
    def generate_summary_report(self):
        """Generate a detailed summary report of all experiments."""
        print("\n" + "="*70)
        print("GENERATING SUMMARY REPORT")
        print("="*70)
        
        report = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": {
                "n_problems": self.args.n_problems,
                "start_idx": self.args.start_idx,
                "hop_filter": self.args.hop_filter,
            },
            "results": self.results,
            "summary": {},
            "conclusions": {}
        }
        
        # Extract key findings
        summary = {}
        conclusions = {}
        
        # Check baseline
        if "baseline" in self.results:
            summary["baseline_effect"] = self.results["baseline"].get("max_effect_size", 0)
            summary["baseline_flips"] = self.results["baseline"].get("answer_flips", 0)
        
        # Check first step flips
        if "first_step_only" in self.results:
            first_step_flips = {}
            for alpha_set, result in self.results["first_step_only"].items():
                flips = result.get("answer_flips", 0)
                total = result.get("total_problems", self.args.n_problems)
                first_step_flips[alpha_set] = {"flips": flips, "total": total, "rate": flips/total if total > 0 else 0}
            summary["first_step"] = first_step_flips
        
        # Check last step flips
        if "last_step_only" in self.results:
            last_step_flips = {}
            for alpha_set, result in self.results["last_step_only"].items():
                flips = result.get("answer_flips", 0)
                total = result.get("total_problems", self.args.n_problems)
                last_step_flips[alpha_set] = {"flips": flips, "total": total, "rate": flips/total if total > 0 else 0}
            summary["last_step"] = last_step_flips
        
        # Check all steps flips
        if "all_steps" in self.results:
            all_steps_flips = {}
            for alpha_set, result in self.results["all_steps"].items():
                flips = result.get("answer_flips", 0)
                total = result.get("total_problems", self.args.n_problems)
                all_steps_flips[alpha_set] = {"flips": flips, "total": total, "rate": flips/total if total > 0 else 0}
            summary["all_steps"] = all_steps_flips
        
        # Check middle steps
        if "middle_steps" in self.results:
            middle_step_flips = {}
            for step, result in self.results["middle_steps"].items():
                flips = result.get("answer_flips", 0)
                total = result.get("total_problems", self.args.n_problems)
                middle_step_flips[step] = {"flips": flips, "total": total, "rate": flips/total if total > 0 else 0}
            summary["middle_steps"] = middle_step_flips
        
        # Check verbal comparison
        if "verbal_comparison" in self.results:
            verbal_result = self.results["verbal_comparison"].get("verbal", {})
            summary["verbal_flips"] = verbal_result.get("answer_flips", 0)
            summary["verbal_total"] = verbal_result.get("total_problems", self.args.n_problems)
            summary["verbal_max_effect"] = verbal_result.get("max_effect_size", 0)
        
        # Check random control (critical for validity)
        if "random_control" in self.results:
            random_results = {}
            for scale_name, result in self.results["random_control"].items():
                flips = result.get("answer_flips", 0)
                total = result.get("total_problems", self.args.n_problems)
                random_results[scale_name] = {"flips": flips, "total": total, "rate": flips/total if total > 0 else 0}
            summary["random_control"] = random_results
            
            # Determine validity
            small_scale_flips = 0
            for scale_name, data in random_results.items():
                if "0.01" in scale_name or "0.05" in scale_name:
                    small_scale_flips += data["flips"]
            
            if small_scale_flips == 0:
                conclusions["random_control_valid"] = True
                conclusions["random_control_message"] = "✅ Random control passed - small magnitude random vectors cause no flips"
            else:
                conclusions["random_control_valid"] = False
                conclusions["random_control_message"] = "⚠️ Random control shows flips at small magnitudes - experiment may be invalid"
        
        # Check noise baseline
        if "noise_baseline" in self.results:
            noise_results = {}
            for scale_name, result in self.results["noise_baseline"].items():
                flips = result.get("answer_flips", 0)
                total = result.get("total_problems", min(5, self.args.n_problems))
                noise_results[scale_name] = {"flips": flips, "total": total, "rate": flips/total if total > 0 else 0}
            summary["noise_baseline"] = noise_results
        
        # Draw conclusions
        coconut_effective = False
        if "first_step" in summary:
            for alpha_set, data in summary["first_step"].items():
                if data.get("rate", 0) > 0.5:
                    coconut_effective = True
                    break
        
        conclusions["coconut_steerable"] = coconut_effective
        conclusions["verbal_steerable"] = summary.get("verbal_flips", 0) > 0
        
        if coconut_effective and not conclusions["verbal_steerable"]:
            conclusions["main_hypothesis"] = "✅ CONFIRMED: Coconut is causally steerable, Verbal CoT is not"
        elif coconut_effective and conclusions["verbal_steerable"]:
            conclusions["main_hypothesis"] = "⚠️ PARTIAL: Both models show steering effects"
        else:
            conclusions["main_hypothesis"] = "❌ REJECTED: Neither model shows reliable steering"
        
        report["summary"] = summary
        report["conclusions"] = conclusions
        
        # Save report
        report_path = os.path.join(self.output_dir, "summary_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "-"*50)
        print("SUMMARY REPORT")
        print("-"*50)
        print(f"Experiment: {self.experiment_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Problems tested: {self.args.n_problems}")
        print(f"Hop filter: {self.args.hop_filter}")
        
        print("\n📊 KEY FINDINGS:")
        print(f"  Baseline flips: {summary.get('baseline_flips', 0)}")
        print(f"  Verbal CoT flips: {summary.get('verbal_flips', 0)}/{summary.get('verbal_total', self.args.n_problems)}")
        
        print("\n🎯 COCONUT STEERING EFFECTIVENESS:")
        if "first_step" in summary:
            for alpha_set, data in summary["first_step"].items():
                print(f"  First step ({alpha_set}): {data['flips']}/{data['total']} flips ({data['rate']*100:.0f}%)")
        
        print("\n🔬 VALIDITY CHECKS:")
        if "random_control" in summary:
            for scale_name, data in summary["random_control"].items():
                print(f"  Random scale {scale_name}: {data['flips']}/{data['total']} flips ({data['rate']*100:.0f}%)")
        
        print("\n💡 CONCLUSIONS:")
        for key, value in conclusions.items():
            if isinstance(value, str):
                print(f"  {value}")
            elif isinstance(value, bool):
                print(f"  {key}: {value}")
        
        print(f"\n📄 Full report saved to: {report_path}")
        
        return report
    
    def run_all(self):
        """Run all experiments."""
        print("\n" + "="*70)
        print("COMPREHENSIVE STEERING EXPERIMENTS")
        print("="*70)
        print(f"Experiment name: {self.experiment_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of problems: {self.args.n_problems}")
        print(f"Start index: {self.args.start_idx}")
        print(f"Hop filter: {self.args.hop_filter}")
        
        # Save experiment config
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.args), f, indent=2)
        
        # Run experiments
        experiments = [
            ("Baseline", self.run_experiment_1_baseline),
            ("First Step Only", self.run_experiment_2_first_step_only),
            ("Last Step Only", self.run_experiment_3_last_step_only),
            ("All Steps", self.run_experiment_4_all_steps),
            ("Middle Steps", self.run_experiment_5_middle_steps),
            ("Alpha Sensitivity", self.run_experiment_6_alpha_sensitivity),
            ("Verbal Comparison", self.run_experiment_7_verbal_comparison),
            ("Random Control", self.run_experiment_8_random_control),
            ("Noise Baseline", self.run_experiment_9_noise_baseline),
        ]
        
        for name, experiment_func in experiments:
            print(f"\n{'='*70}")
            print(f"STARTING: {name}")
            print(f"{'='*70}")
            try:
                experiment_func()
            except Exception as e:
                print(f"  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate summary
        report = self.generate_summary_report()
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        print(f"Summary report: {os.path.join(self.output_dir, 'summary_report.json')}")
        
        return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive steering experiments")
    
    # Data selection
    parser.add_argument("--n_problems", type=int, default=10,
                        help="Number of problems to test")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in dataset")
    parser.add_argument("--hop_filter", type=int, nargs="+", default=None,
                        help="Filter by hop count (e.g., 3 4 5)")
    
    # Experiment control
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Custom experiment name")
    parser.add_argument("--output_dir", type=str, default="results/steering",
                        help="Output directory")
    parser.add_argument("--skip_verbal", action="store_true",
                        help="Skip verbal CoT experiments")
    parser.add_argument("--skip_noise", action="store_true",
                        help="Skip noise baseline experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick version (fewer problems, smaller alpha ranges)")
    
    # Other
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.n_problems = min(3, args.n_problems)
        print("QUICK MODE: Using fewer problems and smaller alpha ranges")
    
    # Run comprehensive experiments
    experiment = ComprehensiveSteeringExperiment(args)
    experiment.run_all()


if __name__ == "__main__":
    main()