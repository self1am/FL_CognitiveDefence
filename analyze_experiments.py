#!/usr/bin/env python3
"""
Post-experiment analysis script for production FL experiments
Analyzes results from 100-client experiments and generates comparison reports
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class ExperimentAnalyzer:
    """Analyze experiment results"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def load_experiments(self) -> None:
        """Load all completed experiment logs"""
        print(f"Loading experiments from {self.logs_dir}...")
        
        for log_file in sorted(self.logs_dir.glob("*_complete.json")):
            exp_name = log_file.stem.replace("_complete", "")
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    self.results[exp_name] = data
                    print(f"  ✓ Loaded: {exp_name}")
            except Exception as e:
                print(f"  ✗ Error loading {exp_name}: {e}")
    
    def compute_metrics(self, data: Dict) -> Dict[str, float]:
        """Compute key metrics from experiment data"""
        metrics = {}
        
        # Accuracy metrics
        accuracy = data.get("centralized_accuracy", [])
        if accuracy:
            metrics["final_accuracy"] = accuracy[-1]
            metrics["max_accuracy"] = max(accuracy)
            metrics["min_accuracy"] = min(accuracy)
            metrics["mean_accuracy"] = np.mean(accuracy)
            metrics["accuracy_improvement"] = accuracy[-1] - accuracy[0]
            metrics["rounds_to_90pct"] = next(
                (i for i, acc in enumerate(accuracy) if acc >= 0.90),
                len(accuracy)
            )
        
        # Loss metrics
        loss = data.get("centralized_loss", [])
        if loss:
            metrics["final_loss"] = loss[-1]
            metrics["max_loss"] = max(loss)
            metrics["min_loss"] = min(loss)
            metrics["mean_loss"] = np.mean(loss)
            metrics["loss_improvement"] = loss[0] - loss[-1]
        
        # Anomaly detection
        anomalies = data.get("detected_anomalies", [])
        if anomalies:
            total_anomalies = sum(len(v) for v in anomalies)
            metrics["total_anomalies_detected"] = total_anomalies
            metrics["avg_anomalies_per_round"] = total_anomalies / len(anomalies) if anomalies else 0
        
        # Client metrics
        metrics["total_rounds"] = len(accuracy) if accuracy else 0
        
        return metrics
    
    def print_summary(self) -> None:
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("PRODUCTION EXPERIMENT ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Experiments: {len(self.results)}")
        print("")
        
        # Compute metrics for each experiment
        all_metrics = {}
        for exp_name, data in self.results.items():
            metrics = self.compute_metrics(data)
            all_metrics[exp_name] = metrics
        
        # Print detailed results
        for exp_name, metrics in all_metrics.items():
            print("-" * 80)
            print(f"Experiment: {exp_name}")
            print("-" * 80)
            
            if metrics:
                print(f"  Rounds Executed:      {metrics.get('total_rounds', 0)}")
                print(f"  Final Accuracy:       {metrics.get('final_accuracy', 0):.4f} ({metrics.get('final_accuracy', 0)*100:.2f}%)")
                print(f"  Max Accuracy:         {metrics.get('max_accuracy', 0):.4f}")
                print(f"  Accuracy Improvement: +{metrics.get('accuracy_improvement', 0):.4f}")
                print(f"  Final Loss:           {metrics.get('final_loss', 0):.6f}")
                print(f"  Loss Improvement:     {metrics.get('loss_improvement', 0):.6f}")
                
                if metrics.get('rounds_to_90pct') is not None:
                    rounds_to_goal = metrics.get('rounds_to_90pct')
                    if rounds_to_goal < metrics.get('total_rounds', float('inf')):
                        print(f"  Rounds to 90% Acc:    {rounds_to_goal}")
                    else:
                        print(f"  Rounds to 90% Acc:    Not reached")
                
                if metrics.get('total_anomalies_detected'):
                    print(f"  Anomalies Detected:   {metrics.get('total_anomalies_detected', 0)}")
                    print(f"  Avg per Round:        {metrics.get('avg_anomalies_per_round', 0):.2f}")
            else:
                print("  No metrics available")
            print("")
        
        # Comparison table
        if len(all_metrics) > 1:
            print("\n" + "=" * 80)
            print("COMPARISON TABLE")
            print("=" * 80)
            
            print(f"{'Experiment':<40} {'Final Acc':<12} {'Final Loss':<12} {'Rounds':<10}")
            print("-" * 80)
            
            for exp_name, metrics in all_metrics.items():
                acc = metrics.get('final_accuracy', 0)
                loss = metrics.get('final_loss', 0)
                rounds = metrics.get('total_rounds', 0)
                print(f"{exp_name:<40} {acc:>10.4f}  {loss:>10.6f}  {rounds:>8}")
            
            print("\n" + "-" * 80)
            print("RANKINGS")
            print("-" * 80)
            
            # Best accuracy
            best_acc_exp = max(all_metrics.items(), key=lambda x: x[1].get('final_accuracy', 0))
            print(f"Best Final Accuracy:    {best_acc_exp[0]} ({best_acc_exp[1].get('final_accuracy', 0):.4f})")
            
            # Best loss
            best_loss_exp = min(all_metrics.items(), key=lambda x: x[1].get('final_loss', float('inf')))
            print(f"Best Final Loss:        {best_loss_exp[0]} ({best_loss_exp[1].get('final_loss', 0):.6f})")
            
            # Fastest convergence
            fastest_exp = min(all_metrics.items(), key=lambda x: x[1].get('rounds_to_90pct', float('inf')))
            print(f"Fastest to 90%:         {fastest_exp[0]} ({fastest_exp[1].get('rounds_to_90pct', float('inf'))} rounds)")
    
    def generate_report(self, output_file: str = "experiment_analysis_report.txt") -> None:
        """Generate detailed text report"""
        print(f"\nGenerating detailed report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PRODUCTION EXPERIMENT ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {len(self.results)}\n\n")
            
            # Detailed results for each experiment
            for exp_name, data in self.results.items():
                metrics = self.compute_metrics(data)
                
                f.write("\n" + "-" * 80 + "\n")
                f.write(f"Experiment: {exp_name}\n")
                f.write("-" * 80 + "\n")
                
                # Write metrics
                f.write("\nKey Metrics:\n")
                for key, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        f.write(f"  {key:.<40} {value:>15.6f}\n")
                    else:
                        f.write(f"  {key:.<40} {value:>15}\n")
                
                # Write accuracy curve
                accuracy = data.get("centralized_accuracy", [])
                if accuracy:
                    f.write("\nAccuracy Progression:\n")
                    for round_num, acc in enumerate(accuracy):
                        f.write(f"  Round {round_num:3d}: {acc:.4f} ({acc*100:6.2f}%)\n")
                
                # Write loss curve
                loss = data.get("centralized_loss", [])
                if loss:
                    f.write("\nLoss Progression:\n")
                    for round_num, l in enumerate(loss):
                        f.write(f"  Round {round_num:3d}: {l:.6f}\n")
        
        print(f"✓ Report saved to {output_file}")
    
    def export_csv(self, output_file: str = "experiment_analysis.csv") -> None:
        """Export results to CSV"""
        print(f"\nExporting to CSV: {output_file}")
        
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            headers = ["Experiment Name", "Final Accuracy", "Final Loss", "Total Rounds", 
                      "Max Accuracy", "Accuracy Improvement", "Anomalies Detected"]
            writer.writerow(headers)
            
            # Data rows
            for exp_name, data in self.results.items():
                metrics = self.compute_metrics(data)
                writer.writerow([
                    exp_name,
                    f"{metrics.get('final_accuracy', 0):.6f}",
                    f"{metrics.get('final_loss', 0):.6f}",
                    metrics.get('total_rounds', 0),
                    f"{metrics.get('max_accuracy', 0):.6f}",
                    f"{metrics.get('accuracy_improvement', 0):.6f}",
                    metrics.get('total_anomalies_detected', 0),
                ])
        
        print(f"✓ CSV exported to {output_file}")

def main():
    """Main analysis function"""
    analyzer = ExperimentAnalyzer()
    
    # Load experiments
    analyzer.load_experiments()
    
    if not analyzer.results:
        print("No experiments found!")
        return 1
    
    # Print summary
    analyzer.print_summary()
    
    # Generate report
    analyzer.generate_report("experiment_analysis_report.txt")
    
    # Export CSV
    analyzer.export_csv("experiment_analysis.csv")
    
    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
