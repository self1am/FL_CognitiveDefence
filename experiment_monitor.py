#!/usr/bin/env python3
"""
Real-time experiment monitoring with timeout detection
"""
import subprocess
import threading
import time
import psutil
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MONITOR - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class ExperimentMonitor:
    def __init__(self, log_file, check_interval=5):
        self.log_file = log_file
        self.check_interval = check_interval
        self.last_log_position = 0
        self.last_activity_time = time.time()
        self.timeout_threshold = 300  # 5 minutes
        self.running = True
        
    def check_log_progress(self):
        """Detect if experiment is progressing"""
        try:
            with open(self.log_file, 'r') as f:
                current_position = f.seek(0, 2)  # Go to end
                position = current_position
                
            if position > self.last_log_position:
                self.last_activity_time = time.time()
                self.last_log_position = position
                logger.info(f"✓ Log activity detected (size: {position} bytes)")
                return True
            else:
                elapsed = time.time() - self.last_activity_time
                logger.warning(f"⚠ NO LOG UPDATE for {elapsed:.0f}s")
                if elapsed > self.timeout_threshold:
                    logger.error(f"🔴 TIMEOUT DETECTED! No activity for {elapsed:.0f}s")
                return False
        except Exception as e:
            logger.error(f"Error checking log: {e}")
            return False
    
    def check_resources(self):
        """Monitor system resources"""
        try:
            # Memory
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk = psutil.disk_usage('/')
            
            logger.info(
                f"📊 Memory: {mem.percent}% (available: {mem.available/1024**3:.1f}GB) | "
                f"Swap: {swap.percent}% ({swap.used/1024**3:.1f}GB/{swap.total/1024**3:.1f}GB) | "
                f"Disk: {disk.percent}% ({disk.free/1024**3:.1f}GB free)"
            )
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            logger.info(f"CPU: {cpu_percent}%")
            
            # Check if critical
            if mem.percent > 90:
                logger.error("🔴 CRITICAL: Memory pressure > 90%!")
            if swap.percent > 50:
                logger.error("🔴 CRITICAL: Swap usage > 50%!")
            if disk.percent > 90:
                logger.error("🔴 CRITICAL: Disk usage > 90%!")
                
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
    
    def monitor_processes(self):
        """Track Python processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'num_threads']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmd = ' '.join(proc.info['cmdline'][:2]) if proc.info['cmdline'] else 'N/A'
                        logger.debug(
                            f"PID {proc.info['pid']}: {proc.info['num_threads']} threads | {cmd}"
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.error(f"Error monitoring processes: {e}")
    
    def run(self):
        """Main monitoring loop"""
        logger.info("Starting experiment monitor...")
        while self.running:
            self.check_log_progress()
            self.check_resources()
            self.monitor_processes()
            time.sleep(self.check_interval)

if __name__ == '__main__':
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'baseline_100_clients.log'
    monitor = ExperimentMonitor(log_file)
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
        monitor.running = False
