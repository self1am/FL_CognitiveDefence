# src/orchestration/client_orchestrator.py
import subprocess
import time
import psutil
import json
import os
import signal
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
import queue
from pathlib import Path

from ..utils.logging_utils import ExperimentLogger
from ..utils.config import ExperimentConfig, ClientConfig, AttackConfig

@dataclass
class ClientProcess:
    """Information about a running client process"""
    client_id: int
    process: subprocess.Popen
    config: Dict[str, Any]
    resource_usage: Dict[str, float] = None
    status: str = "running"  # running, completed, failed
    start_time: float = 0

class ResourceMonitor:
    """Monitor system resources to manage client processes"""
    
    def __init__(self, max_memory_mb: int = 6000, max_cpu_percent: int = 80):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.monitoring = False
        self.resources = {}
        self.lock = threading.Lock()
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current system resource usage"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        return {
            'memory_mb': memory.used / (1024 * 1024),
            'memory_percent': memory.percent,
            'cpu_percent': cpu,
            'available_memory_mb': memory.available / (1024 * 1024)
        }
    
    def can_spawn_client(self, estimated_memory_mb: int = 500) -> bool:
        """Check if system can handle another client"""
        current = self.get_current_usage()
        
        # Require at least estimated_memory_mb available, but be more lenient than before
        # Allow spawning if we have at least 300MB (reduced from 500MB to handle memory pressure)
        memory_ok = (current['available_memory_mb'] > 300)
        
        # Don't block on CPU if we have available memory
        # CPU will naturally throttle process spawning
        cpu_ok = True
        
        return memory_ok and cpu_ok
    
    def start_monitoring(self, client_processes: Dict[int, ClientProcess]):
        """Start resource monitoring thread"""
        self.monitoring = True
        self.client_processes = client_processes
        
        def monitor_loop():
            while self.monitoring:
                try:
                    current_usage = self.get_current_usage()
                    with self.lock:
                        self.resources = current_usage
                    
                    # Check individual client processes
                    for client_id, client_proc in self.client_processes.items():
                        if client_proc.process and client_proc.process.poll() is None:
                            try:
                                proc = psutil.Process(client_proc.process.pid)
                                client_proc.resource_usage = {
                                    'memory_mb': proc.memory_info().rss / (1024 * 1024),
                                    'cpu_percent': proc.cpu_percent()
                                }
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                client_proc.status = "failed"
                    
                    time.sleep(5)  # Monitor every 5 seconds
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(5)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)

class ClientOrchestrator:
    """Orchestrate multiple client processes"""
    
    def __init__(self, 
                 server_address: str,
                 experiment_config: ExperimentConfig,
                 logger: Optional[ExperimentLogger] = None,
                 max_memory_mb: int = 6000):
        
        self.server_address = server_address
        self.experiment_config = experiment_config
        self.logger = logger
        self.client_processes: Dict[int, ClientProcess] = {}
        self.resource_monitor = ResourceMonitor(max_memory_mb=max_memory_mb)
        self.client_script_path = "src.clients.client_runner"
        
        # Convert 0.0.0.0 to localhost for client connections
        # Clients need a connectable address, not the bind address
        if server_address.startswith("0.0.0.0"):
            self.client_connect_address = server_address.replace("0.0.0.0", "localhost")
            if self.logger:
                self.logger.logger.info(f"Server listening on {server_address}, clients will connect to {self.client_connect_address}")
        else:
            self.client_connect_address = server_address
        
    def generate_client_config(self, client_id: int, attack_config: Optional[AttackConfig] = None) -> Dict[str, Any]:
        """Generate configuration for a specific client"""
        config = {
            'client_id': client_id,
            'server_address': self.client_connect_address,  # Use connectable address
            'experiment_name': self.experiment_config.experiment_name,
            'seed': self.experiment_config.seed + client_id,  # Unique seed per client
            'batch_size': 32,
            'epochs': 2,
            'learning_rate': 0.001
        }
        
        if attack_config and attack_config.enabled:
            config.update({
                'attack_enabled': True,
                'attack_type': attack_config.attack_type,
                'attack_intensity': attack_config.intensity,
            })
        else:
            config['attack_enabled'] = False
        
        return config
    
    def spawn_client(self, client_id: int, config: Dict[str, Any], delay: float = 0) -> Optional[ClientProcess]:
        """Spawn a single client process"""
        if delay > 0:
            time.sleep(delay)
        
        # Check resources before spawning
        if not self.resource_monitor.can_spawn_client():
            if self.logger:
                self.logger.logger.warning(f"Cannot spawn client {client_id} - insufficient resources")
            return None
        
        try:
            # Prepare command
            cmd = [
                "python", "-m", self.client_script_path,
                "--config", json.dumps(config)
            ]
            
            # Create log file for client output
            client_log_file = f"logs/client_{client_id}.log"
            Path("logs").mkdir(exist_ok=True)
            
            # Start process with output redirection
            with open(client_log_file, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
            
            client_process = ClientProcess(
                client_id=client_id,
                process=process,
                config=config,
                start_time=time.time()
            )
            
            self.client_processes[client_id] = client_process
            
            if self.logger:
                self.logger.logger.info(f"Spawned client {client_id} with PID {process.pid} (logs: {client_log_file})")
            
            return client_process
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Failed to spawn client {client_id}: {e}")
            return None
    
    def spawn_clients_batch(self, 
                           client_configs: List[Tuple[int, Dict[str, Any]]], 
                           batch_size: int = 3,
                           spawn_delay: float = 2.0) -> List[ClientProcess]:
        """Spawn clients in batches to manage resources"""
        spawned_clients = []
        
        for i in range(0, len(client_configs), batch_size):
            batch = client_configs[i:i + batch_size]
            
            # Spawn batch concurrently
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for j, (client_id, config) in enumerate(batch):
                    delay = j * spawn_delay  # Stagger within batch
                    future = executor.submit(self.spawn_client, client_id, config, delay)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    client_process = future.result()
                    if client_process:
                        spawned_clients.append(client_process)
            
            # Wait between batches
            if i + batch_size < len(client_configs):
                time.sleep(spawn_delay * batch_size)
        
        return spawned_clients
    
    def run_experiment(self, 
                      num_clients: int = 10,
                      attack_configs: Optional[Dict[int, AttackConfig]] = None,
                      batch_size: int = 3,
                      server_process = None) -> Dict[str, Any]:
        """Run complete multi-client experiment
        
        Args:
            num_clients: Number of clients to spawn
            attack_configs: Attack configurations per client
            batch_size: Number of clients to spawn per batch
            server_process: Reference to server process (multiprocessing.Process)
        """
        
        if self.logger:
            self.logger.logger.info(f"Starting experiment with {num_clients} clients")
        
        # Generate client configurations
        client_configs = []
        for client_id in range(num_clients):
            attack_config = attack_configs.get(client_id) if attack_configs else None
            config = self.generate_client_config(client_id, attack_config)
            client_configs.append((client_id, config))
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring(self.client_processes)
        
        # Spawn clients
        start_time = time.time()
        spawned_clients = self.spawn_clients_batch(client_configs, batch_size)
        
        if self.logger:
            self.logger.logger.info(f"Successfully spawned {len(spawned_clients)}/{num_clients} clients")
        
        # Monitor client processes
        self.monitor_clients()
        
        # Wait for server completion instead of client completion
        if server_process:
            if self.logger:
                self.logger.logger.info("Waiting for server to complete training rounds...")
            server_process.join()  # Block until server process exits
            if self.logger:
                self.logger.logger.info("✅ Server completed all training rounds")
        else:
            # Fallback: wait for clients (shouldn't happen in normal flow)
            if self.logger:
                self.logger.logger.warning("No server process provided, waiting for clients instead")
            self.wait_for_completion()
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        
        # Terminate all clients gracefully
        self.terminate_all_clients()
        
        # Collect results
        experiment_duration = time.time() - start_time
        results = self.collect_results()
        
        experiment_summary = {
            'experiment_name': self.experiment_config.experiment_name,
            'total_clients': num_clients,
            'successful_clients': len([c for c in self.client_processes.values() if c.status == "completed"]),
            'failed_clients': len([c for c in self.client_processes.values() if c.status == "failed"]),
            'duration_seconds': experiment_duration,
            'results': results
        }
        
        if self.logger:
            self.logger.logger.info(f"Experiment completed: {experiment_summary}")
        
        return experiment_summary
    
    def monitor_clients(self):
        """Monitor client process status and connectivity"""
        def monitoring_loop():
            while any(proc.status == "running" for proc in self.client_processes.values()):
                for client_id, client_proc in self.client_processes.items():
                    if client_proc.process and client_proc.status == "running":
                        poll_result = client_proc.process.poll()
                        if poll_result is not None:
                            if poll_result == 0:
                                client_proc.status = "completed"
                                if self.logger:
                                    self.logger.logger.info(f"Client {client_id} completed successfully")
                            else:
                                client_proc.status = "failed"
                                if self.logger:
                                    self.logger.logger.error(f"Client {client_id} failed with exit code {poll_result}")
                                    # Read error logs
                                    try:
                                        with open(f"logs/client_{client_id}.log", 'r') as f:
                                            error_output = f.read()
                                            if error_output:
                                                self.logger.logger.error(f"Client {client_id} output:\n{error_output[-500:]}")  # Last 500 chars
                                    except:
                                        pass
                
                time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def wait_for_completion(self, timeout: float = None):  # None = wait indefinitely
        """Wait for all clients to complete their training
        
        Note: Clients don't exit naturally - they stay connected to server.
        This waits for either:
        1. A timeout (if specified)
        2. User interruption (Ctrl+C)
        3. Server completion (detected externally)
        """
        if timeout is None:
            timeout = float('inf')
        
        start_time = time.time()
        active_clients = len([c for c in self.client_processes.values() if c.status == "running"])
        
        if self.logger:
            self.logger.logger.info(f"Waiting for {active_clients} clients to train...")
            self.logger.logger.info(f"Training is happening in the background. Timeout: {timeout if timeout != float('inf') else 'None (indefinite)'}")
        
        try:
            while time.time() - start_time < timeout:
                active = [
                    cid for cid, proc in self.client_processes.items() 
                    if proc.status == "running" and proc.process.poll() is None
                ]
                
                if not active:
                    if self.logger:
                        self.logger.logger.info("All clients have finished or disconnected")
                    break
                
                # Update status every 30 seconds instead of 10
                time.sleep(30)
                
        except KeyboardInterrupt:
            if self.logger:
                self.logger.logger.info("Received interrupt signal. Terminating all clients...")
            self.terminate_all_clients()
            raise
        finally:
            # Ensure all clients are cleaned up
            self.terminate_all_clients()
    
    def terminate_all_clients(self):
        """Terminate all client processes"""
        for client_id, client_proc in self.client_processes.items():
            if client_proc.process and client_proc.process.poll() is None:
                try:
                    client_proc.process.terminate()
                    client_proc.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    client_proc.process.kill()
                except Exception as e:
                    if self.logger:
                        self.logger.logger.error(f"Error terminating client {client_id}: {e}")
    
    def collect_results(self) -> Dict[str, Any]:
        """Collect results from all clients"""
        results = {
            'client_logs': {},
            'resource_usage': {},
            'status_summary': {}
        }
        
        for client_id, client_proc in self.client_processes.items():
            results['status_summary'][client_id] = client_proc.status
            results['resource_usage'][client_id] = client_proc.resource_usage
            
            # Try to read client output
            try:
                if client_proc.process:
                    stdout, stderr = client_proc.process.communicate(timeout=5)
                    results['client_logs'][client_id] = {
                        'stdout': stdout,
                        'stderr': stderr
                    }
            except subprocess.TimeoutExpired:
                results['client_logs'][client_id] = {'error': 'timeout_reading_output'}
            except Exception as e:
                results['client_logs'][client_id] = {'error': str(e)}
        
        return results