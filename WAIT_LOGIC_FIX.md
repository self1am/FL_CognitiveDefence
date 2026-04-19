# Wait Logic Fix - Server-Driven Synchronization

## Problem
Orchestrator was stuck in "Waiting for 100 clients" loop forever after clients were spawned.

```
2026-02-06 10:11:44,711 - production_100_clients_cognitive_defence - INFO - Waiting for 100 clients: [0, 1, 2, ..., 99]
2026-02-06 10:11:54,711 - production_100_clients_cognitive_defence - INFO - Waiting for 100 clients: [0, 1, 2, ..., 99]
... repeats forever
```

## Root Cause
The wait logic was checking if **client processes exit**, but:
- Clients call `fl.client.start_client()` which **blocks indefinitely** waiting for server
- Clients never naturally exit - they stay connected throughout all training rounds
- The `wait_for_completion()` checked `process.poll()` which never returns a value
- **Result:** Infinite wait loop

## Solution
Changed synchronization model from **"wait for clients to exit"** to **"wait for server to complete"**:

1. **Pass server process to orchestrator**
   - Server process reference now passed to `run_experiment(server_process=...)`

2. **Wait for server completion**
   - Instead of: `wait_for_completion()` (waits for client exit)
   - Now: `server_process.join()` (waits for server process to exit)
   - Server exits automatically after completing all training rounds

3. **Clients handled gracefully**
   - After server exits, orchestrator terminates all client processes
   - Clients don't need to exit naturally

## Code Changes

### `experiment_runner.py`
```python
experiment_results = orchestrator.run_experiment(
    num_clients=num_clients,
    attack_configs=attack_configs,
    batch_size=batch_size,
    server_process=server_process  # NEW: pass server process
)
```

### `client_orchestrator.py`
```python
def run_experiment(self, 
                  num_clients: int = 10,
                  attack_configs: Optional[Dict[int, AttackConfig]] = None,
                  batch_size: int = 3,
                  server_process = None) -> Dict[str, Any]:
    
    # Spawn all clients
    spawned_clients = self.spawn_clients_batch(client_configs, batch_size)
    
    # WAIT FOR SERVER TO COMPLETE (not clients)
    if server_process:
        server_process.join()  # Blocks until server exits
        self.logger.logger.info("✅ Server completed all training rounds")
    
    # THEN terminate clients
    self.terminate_all_clients()
```

## Additional Improvements

### Client Output Logging
- Each client now logs to `logs/client_*.log` file
- Captures client errors and diagnostics
- Helps debug client connectivity issues

### Better Monitoring
- Enhanced `monitor_clients()` to read error logs when clients fail
- Shows last 500 chars of client output on error
- Better visibility into what's happening

### Relaxed Wait Timeout
- Old: Fixed 30-minute timeout
- New: Waits indefinitely for server (proper synchronization)
- Only interrupted by user (Ctrl+C) or server completion

## Expected Behavior

**Before (broken):**
```
All 100 clients spawned
Waiting for 100 clients: [0, 1, 2, ..., 99]  ← STUCK FOREVER
```

**After (fixed):**
```
All 100 clients spawned
Waiting for server to complete training rounds...
[Server runs rounds 1-40 while clients participate]
✅ Server completed all training rounds
Terminating all clients...
Experiment completed: {...}
```

## Testing
The fix ensures that:
1. Clients spawn and connect to server
2. Orchestrator waits for server completion (not client exit)
3. Experiment progresses through all rounds
4. Proper cleanup occurs when finished
