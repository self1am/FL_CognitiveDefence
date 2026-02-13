#!/usr/bin/env python3
"""
Safe experiment runner for GCP browser SSH environments.
Handles timeouts, disconnections, and monitors progress independently.
"""
import os
import sys
import json
import time
import logging
import subprocess
import signal
import atexit
from datetime import datetime
from pathlib import Path

# Setup logging that goes to both console AND file
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f'experiment_safe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SAFE_RUN - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class SafeExperimentRunner:
    def __init__(self, script_name='run_server_with_eval.py', config_file=None):
        self.script = script_name
        self.config = config_file
        self.process = None
        self.process_file = Path('experiment.pid')
        self.start_time = None
        self.last_log_size = 0
        
    def on_exit(self):
        """Cleanup on exit"""
        if self.process and self.process.poll() is None:
            logger.warning("Process still running - detaching (will continue in background)")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate, forcing kill")
                self.process.kill()
        
        logger.info(f"Logs saved to: {log_file}")
    
    def handle_signal(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("Received Ctrl+C - experiment continues in background")
        logger.info(f"To monitor: tail -f {log_file}")
        logger.info(f"To check status: ps -p $(cat {self.process_file})")
        sys.exit(0)
    
    def run(self):
        """Start experiment with proper error handling"""
        logger.info("=" * 80)
        logger.info("SAFE EXPERIMENT RUNNER - Browser SSH Compatible")
        logger.info("=" * 80)
        logger.info(f"Config: {self.config}")
        logger.info(f"Script: {self.script}")
        logger.info(f"Process ID file: {self.process_file}")
        logger.info(f"Main log output: {log_file}")
        logger.info("")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        atexit.register(self.on_exit)
        
        # Build command
        cmd = [sys.executable, self.script]
        if self.config:
            cmd.extend(['--config', self.config])
        
        logger.info(f"Starting: {' '.join(cmd)}")
        logger.info("NOTE: You can close this browser tab - experiment continues on VM")
        logger.info("      To reconnect: tail -f " + str(log_file))
        logger.info("")
        
        if log_file.exists():
            self.last_log_size = log_file.stat().st_size
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Save PID for external monitoring
            self.process_file.write_text(str(self.process.pid))
            logger.info(f"Process started with PID: {self.process.pid}")
            
            self.start_time = time.time()
            blank_lines = 0
            
            # Stream output
            while self.process.poll() is None:
                try:
                    line = self.process.stdout.readline()
                    if line:
                        print(line.rstrip())  # Print to console immediately
                        blank_lines = 0
                    else:
                        blank_lines += 1
                        # If no output for a while, print status
                        if blank_lines > 100:  # Every ~10 seconds
                            elapsed = time.time() - self.start_time
                            logger.debug(f"[{elapsed:.0f}s elapsed] Still running...")
                            blank_lines = 0
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error reading output: {e}")
                    break
            
            # Get final output
            remaining = self.process.stdout.read()
            if remaining:
                print(remaining)
            
            ret_code = self.process.returncode
            elapsed = time.time() - self.start_time
            
            if ret_code == 0:
                logger.info(f"✓ Experiment completed successfully in {elapsed:.0f}s")
            else:
                logger.error(f"✗ Experiment failed with exit code {ret_code} after {elapsed:.0f}s")
            
            return ret_code
            
        except KeyboardInterrupt:
            logger.info("User interrupted (Ctrl+C)")
            raise
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Safe experiment runner for GCP browser SSH'
    )
    parser.add_argument(
        '--script',
        default='run_server_with_eval.py',
        help='Python script to run'
    )
    parser.add_argument(
        '--config',
        help='Config file path'
    )
    
    args = parser.parse_args()
    
    runner = SafeExperimentRunner(script_name=args.script, config_file=args.config)
    exit_code = runner.run()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
