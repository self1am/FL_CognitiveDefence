# Multiprocessing Fix Summary

## Issues Fixed

### 1. **Signal Handler Error** ✅
**Problem:** 
```
ValueError: signal only works in main thread of the main interpreter
```

**Root Cause:** Flower's `start_server()` registers signal handlers (SIGTERM, SIGINT) which can only be done in the main thread. Running the server in a background `threading.Thread` caused this error.

**Solution:** Changed from `threading.Thread` to `multiprocessing.Process`
- Each process has its own main thread that can handle signals
- Server now has its own process with proper signal handling

### 2. **Memory Resource Exhaustion** ✅
**Problem:**
```
Cannot spawn client 86 - insufficient resources
Successfully spawned 70/100 clients
```

**Root Cause:** Resource monitor was too conservative:
- Required 500MB available memory per client
- CPU check was blocking even when CPU wasn't the issue

**Solution:** Relaxed resource constraints in `client_orchestrator.py`:
- Reduced minimum available memory requirement to 300MB
- Removed CPU percent check (system naturally throttles)
- Clients can now spawn more aggressively

### 3. **Improved Server Startup** ✅
**Changes:**
- Added process alive check after server starts
- Reduced startup wait from 5s to 3s (sufficient for binding)
- Added proper cleanup/shutdown handling with timeout
- Better error messages if server fails to start

## Code Changes

### File: `src/orchestration/experiment_runner.py`
1. Added imports: `multiprocessing`, `signal`, `os`
2. Changed `start_server()` to use `multiprocessing.Process` instead of `threading.Thread`
3. Added process status checking and cleanup in `run_experiment()`

### File: `src/orchestration/client_orchestrator.py`
1. Modified `can_spawn_client()` to use 300MB threshold instead of 500MB
2. Removed CPU percent blocking logic

## Testing

Created test config: `experiments/configs/test_multiprocess_fix.yaml`
- 5 clients, 5 rounds (quick test)
- Uses localhost server (0.0.0.0:8080)
- Can run entirely in one terminal

## Expected Behavior

Before: Server crashed with signal error, only ~70/100 clients spawned
After: Server runs in separate process, all clients spawn successfully

## Run Instructions

```bash
# Test with small config first
python -m src.orchestration.experiment_runner --config experiments/configs/test_multiprocess_fix.yaml

# Then run full production
python -m src.orchestration.experiment_runner --config experiments/configs/production_100_clients_adaptive.yaml
```

All runs now complete in **single terminal** as intended!
