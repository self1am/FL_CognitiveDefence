#!/usr/bin/env python3
"""
Real-time RAM monitoring during FL experiments.
Tracks peak memory, per-process breakdown, and swap usage.
"""
import psutil
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RAM_MONITOR - %(message)s',
    handlers=[
        logging.FileHandler('ram_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class RAMMonitor:
    def __init__(self, sample_interval=5, max_samples=1000):
        self.sample_interval = sample_interval
        self.samples = deque(maxlen=max_samples)  # Keep last 1000 samples (~83 minutes at 5s interval)
        self.peak_memory = 0
        self.peak_timestamp = None
        self.data_file = Path('ram_measurements.json')
        
    def get_process_memory(self):
        """Get memory breakdown by process type"""
        python_procs = []
        ray_procs = []
        other_procs = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
                try:
                    mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                    
                    if 'python' in proc.info['name'].lower():
                        cmd_str = ' '.join(proc.info['cmdline'][:2]) if proc.info['cmdline'] else 'python'
                        python_procs.append({
                            'pid': proc.info['pid'],
                            'cmd': cmd_str,
                            'memory_mb': mem_mb
                        })
                        
                        if 'ray' in cmd_str.lower():
                            ray_procs.append({
                                'pid': proc.info['pid'],
                                'memory_mb': mem_mb
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.debug(f"Error getting process memory: {e}")
        
        return {
            'python_processes': python_procs,
            'ray_processes': ray_procs,
            'num_python': len(python_procs),
            'num_ray': len(ray_procs)
        }
    
    def get_system_memory(self):
        """Get overall system memory usage"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            'used_gb': mem.used / 1024**3,
            'percent': mem.percent,
            'swap_total_gb': swap.total / 1024**3,
            'swap_used_gb': swap.used / 1024**3,
            'swap_percent': swap.percent
        }
    
    def sample(self):
        """Collect one measurement"""
        timestamp = datetime.now()
        sys_mem = self.get_system_memory()
        proc_mem = self.get_process_memory()
        
        sample = {
            'timestamp': timestamp.isoformat(),
            'system': sys_mem,
            'processes': proc_mem
        }
        
        self.samples.append(sample)
        
        # Update peak
        if sys_mem['used_gb'] > self.peak_memory:
            self.peak_memory = sys_mem['used_gb']
            self.peak_timestamp = timestamp.isoformat()
        
        return sample
    
    def print_status(self):
        """Pretty print current status"""
        if not self.samples:
            return
        
        latest = self.samples[-1]
        sys = latest['system']
        procs = latest['processes']
        
        logger.info(
            f"RAM: {sys['used_gb']:.1f}GB/{sys['total_gb']:.1f}GB ({sys['percent']:.1f}%) | "
            f"Swap: {sys['swap_used_gb']:.1f}GB/{sys['swap_total_gb']:.1f}GB ({sys['swap_percent']:.1f}%) | "
            f"Python: {procs['num_python']} procs | Ray: {procs['num_ray']} workers"
        )
    
    def save_data(self):
        """Save all measurements to file"""
        data = {
            'measurements': list(self.samples),
            'peak_memory_gb': self.peak_memory,
            'peak_timestamp': self.peak_timestamp,
            'num_samples': len(self.samples),
            'duration_minutes': (len(self.samples) * self.sample_interval) / 60
        }
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data saved to {self.data_file}")
    
    def run(self):
        """Main monitoring loop"""
        logger.info("Starting RAM monitor...")
        logger.info(f"Sample interval: {self.sample_interval}s")
        
        try:
            while True:
                self.sample()
                self.print_status()
                time.sleep(self.sample_interval)
        except KeyboardInterrupt:
            logger.info("Monitor stopped")
            self.save_data()
            logger.info(f"Peak memory: {self.peak_memory:.1f}GB at {self.peak_timestamp}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor RAM during FL experiments')
    parser.add_argument('--interval', type=int, default=5, help='Sample interval (seconds)')
    args = parser.parse_args()
    
    monitor = RAMMonitor(sample_interval=args.interval)
    monitor.run()
