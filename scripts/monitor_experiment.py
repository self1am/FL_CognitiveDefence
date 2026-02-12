#!/usr/bin/env python3
"""
Experiment monitoring script that displays real-time status
Usage: python scripts/monitor_experiment.py [experiment_name]
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


class ExperimentMonitor:
    def __init__(self, log_dir: str = "logs/experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def find_status_files(self, experiment_name: str = None):
        """Find all active experiment status files"""
        pattern = f"{experiment_name}_status.json" if experiment_name else "*_status.json"
        return list(self.log_dir.glob(pattern))
    
    def read_status(self, status_file: Path):
        """Read status from JSON file"""
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def get_log_tail(self, log_file: str, lines: int = 20):
        """Get last N lines from log file"""
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                return ''.join(all_lines[-lines:])
        except FileNotFoundError:
            return "Log file not found"
    
    def parse_experiment_progress(self, log_file: str):
        """Parse log file to extract progress information"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            total_rounds = None
            current_round = 0
            last_accuracy = None
            last_loss = None
            
            for line in lines:
                # Extract total rounds
                if 'num_rounds=' in line:
                    try:
                        total_rounds = int(line.split('num_rounds=')[1].split(',')[0])
                    except (ValueError, IndexError):
                        pass
                
                # Extract current round
                if '[ROUND' in line:
                    try:
                        current_round = int(line.split('[ROUND')[1].split(']')[0].strip())
                    except (ValueError, IndexError):
                        pass
                
                # Extract accuracy and loss
                if 'Accuracy:' in line:
                    try:
                        last_accuracy = float(line.split('Accuracy:')[1].split()[0])
                    except (ValueError, IndexError):
                        pass
                
                if 'Loss:' in line:
                    try:
                        last_loss = float(line.split('Loss:')[1].split(',')[0].strip())
                    except (ValueError, IndexError):
                        pass
            
            return {
                'total_rounds': total_rounds,
                'current_round': current_round,
                'last_accuracy': last_accuracy,
                'last_loss': last_loss,
            }
        except FileNotFoundError:
            return {}
    
    def format_status(self, status: dict, progress: dict):
        """Format status for display"""
        output = []
        output.append("=" * 80)
        output.append(f"📊 Experiment Status")
        output.append("=" * 80)
        
        # Basic info
        output.append(f"PID:          {status.get('pid', 'N/A')}")
        output.append(f"Status:       {status.get('status', 'unknown').upper()}")
        output.append(f"Config:       {Path(status.get('config_file', 'N/A')).name}")
        
        # Progress
        if progress.get('total_rounds') and progress.get('current_round') is not None:
            progress_pct = (progress['current_round'] / progress['total_rounds']) * 100
            output.append(f"Progress:     Round {progress['current_round']}/{progress['total_rounds']} ({progress_pct:.1f}%)")
        
        # Metrics
        if progress.get('last_accuracy') is not None:
            output.append(f"Last Accuracy: {progress['last_accuracy']:.4f}")
        if progress.get('last_loss') is not None:
            output.append(f"Last Loss:     {progress['last_loss']:.4f}")
        
        # Resources
        if status.get('memory_mb'):
            output.append(f"Memory:       {status['memory_mb']} MB")
        if status.get('cpu_percent'):
            output.append(f"CPU:          {status['cpu_percent']}%")
        
        # Timestamps
        if status.get('last_update'):
            output.append(f"Last Update:  {status['last_update']}")
        
        output.append("=" * 80)
        
        return "\n".join(output)
    
    def monitor(self, experiment_name: str = None, interval: int = 10, show_log_tail: bool = False):
        """Monitor experiment(s) in real-time"""
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                
                status_files = self.find_status_files(experiment_name)
                
                if not status_files:
                    print(f"No active experiments found matching: {experiment_name or 'any'}")
                    print(f"Looking in: {self.log_dir.absolute()}")
                    time.sleep(interval)
                    continue
                
                for status_file in status_files:
                    status = self.read_status(status_file)
                    if not status:
                        continue
                    
                    log_file = status.get('log_file')
                    progress = self.parse_experiment_progress(log_file) if log_file else {}
                    
                    print(self.format_status(status, progress))
                    
                    if show_log_tail and log_file:
                        print("\n📋 Recent Log Entries:")
                        print("-" * 80)
                        print(self.get_log_tail(log_file, lines=15))
                    
                    print("\n")
                
                print(f"[Press Ctrl+C to exit] Refreshing in {interval} seconds...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            sys.exit(0)
    
    def list_experiments(self):
        """List all experiments with status files"""
        status_files = self.find_status_files()
        
        if not status_files:
            print("No experiments found.")
            return
        
        print("=" * 80)
        print("Active/Recent Experiments")
        print("=" * 80)
        
        for status_file in status_files:
            status = self.read_status(status_file)
            if not status:
                continue
            
            exp_name = status_file.stem.replace('_status', '')
            pid = status.get('pid', 'N/A')
            status_str = status.get('status', 'unknown')
            
            # Check if process is still running
            if pid != 'N/A':
                try:
                    os.kill(int(pid), 0)
                    running = "🟢 RUNNING"
                except (OSError, ValueError):
                    running = "🔴 STOPPED"
            else:
                running = "❓ UNKNOWN"
            
            print(f"{exp_name:30s} | PID: {str(pid):8s} | {running:15s} | {status_str}")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monitor federated learning experiments")
    parser.add_argument(
        "experiment_name",
        nargs="?",
        help="Name of experiment to monitor (optional, monitors all if not specified)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments and exit"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--log-dir",
        default="logs/experiments",
        help="Directory containing experiment logs"
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Show recent log entries"
    )
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor(log_dir=args.log_dir)
    
    if args.list:
        monitor.list_experiments()
    else:
        monitor.monitor(
            experiment_name=args.experiment_name,
            interval=args.interval,
            show_log_tail=args.show_logs
        )


if __name__ == "__main__":
    main()
