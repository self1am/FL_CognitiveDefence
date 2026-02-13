#!/usr/bin/env python3
"""
Profile CPU and I/O to find bottleneck
"""
import psutil
import time
import json
import logging
from pathlib import Path
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CPU_PROFILER - %(message)s',
    handlers=[
        logging.FileHandler('cpu_profiler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class CPUProfiler:
    def __init__(self, sample_interval=5):
        self.interval = sample_interval
        self.samples = deque(maxlen=1000)
        
    def sample(self):
        """Collect CPU metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            
            # Per-core usage
            cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.5)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Process CPU
            python_cpu = 0
            ray_cpu = 0
            num_python = 0
            
            for proc in psutil.process_iter(['name', 'cpu_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        num_python += 1
                        cpu = proc.info['cpu_percent'] or 0
                        python_cpu += cpu
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            sample = {
                'timestamp': time.time(),
                'cpu_total': cpu_percent,
                'cpu_per_core': cpu_per_core,
                'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
                'python_procs': num_python,
                'python_total_cpu': python_cpu,
                'disk_read_mb': disk_io.read_bytes / 1024**2 if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / 1024**2 if disk_io else 0,
            }
            
            self.samples.append(sample)
            return sample
        except Exception as e:
            logger.error(f"Error sampling: {e}")
            return None
    
    def print_status(self):
        """Print current status"""
        if not self.samples:
            return
        
        latest = self.samples[-1]
        
        # Calculate per-core load
        cores_above_50 = sum(1 for c in latest['cpu_per_core'] if c > 50)
        cores_above_75 = sum(1 for c in latest['cpu_per_core'] if c > 75)
        
        logger.info(
            f"CPU: {latest['cpu_total']:>5.1f}% | "
            f"Cores >50%: {cores_above_50}/8 | "
            f"Cores >75%: {cores_above_75}/8 | "
            f"Freq: {latest['cpu_freq_mhz']:.0f} MHz | "
            f"Python: {latest['python_procs']} procs ({latest['python_total_cpu']:.1f}%)"
        )
    
    def analyze(self):
        """Analyze collected data"""
        if len(self.samples) < 10:
            return
        
        logger.info("\n" + "="*80)
        logger.info("CPU ANALYSIS")
        logger.info("="*80)
        
        cpus = [s['cpu_total'] for s in self.samples]
        py_cpus = [s['python_total_cpu'] for s in self.samples]
        
        logger.info(f"\nTotal CPU Usage:")
        logger.info(f"  Average: {sum(cpus)/len(cpus):.1f}%")
        logger.info(f"  Peak:    {max(cpus):.1f}%")
        logger.info(f"  Min:     {min(cpus):.1f}%")
        
        logger.info(f"\nPython CPU Usage:")
        logger.info(f"  Average: {sum(py_cpus)/len(py_cpus):.1f}%")
        logger.info(f"  Peak:    {max(py_cpus):.1f}%")
        
        # Check core utilization
        all_cores = []
        for sample in self.samples:
            all_cores.extend(sample['cpu_per_core'])
        
        core_avg = sum(all_cores) / len(all_cores) if all_cores else 0
        logger.info(f"\nPer-Core Average:")
        logger.info(f"  {core_avg:.1f}% per core")
        logger.info(f"  Full utilization would be: {core_avg * 8:.1f}% total")
        
        # Bottleneck detection
        logger.info(f"\n" + "-"*80)
        logger.info("BOTTLENECK ANALYSIS:")
        logger.info("-"*80)
        
        if max(cpus) < 50:
            logger.info("⚠️  CPU < 50% utilized")
            logger.info("   Problem: Not enough parallelism")
            logger.info("   Solution: Increase num_clients or reduce num_cpus per client")
        elif max(cpus) > 90:
            logger.info("⚠️  CPU > 90% utilized")
            logger.info("   Problem: CPU is bottleneck, can't parallelize more")
            logger.info("   Solution: Upgrade to more vCPUs OR reduce clients")
        else:
            logger.info("✅ CPU well utilized (50-90%)")
            logger.info("   Status: Good parallelism")
        
        if max(py_cpus) < 30:
            logger.info("⚠️  Python not using allocated CPU")
            logger.info("   Problem: I/O bound, not CPU bound")
            logger.info("   Solution: Check disk I/O, network latency, or data loading")

    def run(self):
        """Main loop"""
        logger.info("Starting CPU profiler...")
        logger.info(f"Sample interval: {self.interval}s")
        logger.info("Press Ctrl+C to stop and analyze")
        
        try:
            while True:
                sample = self.sample()
                if sample:
                    self.print_status()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
            self.analyze()

if __name__ == '__main__':
    profiler = CPUProfiler(sample_interval=5)
    profiler.run()
