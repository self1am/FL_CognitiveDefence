#!/usr/bin/env python3
"""
Analyze RAMmeasurements collected during an experiment.
Generates summary statistics and visualizations.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

def analyze_ram_log(log_file='ram_measurements.json'):
    """Analyze RAM measurements file"""
    
    if not Path(log_file).exists():
        print(f"❌ No measurements file found: {log_file}")
        print("\nTo create measurements:")
        print("  1. Start experiment in one tmux window")
        print("  2. In another window: python ram_monitor.py")
        print("  3. Monitor will create ram_measurements.json")
        return
    
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return
    
    measurements = data.get('measurements', [])
    if not measurements:
        print("No measurements in file")
        return
    
    print("\n" + "="*80)
    print("RAM MEASUREMENTS ANALYSIS")
    print("="*80)
    print(f"\nMeasurement Duration: {data.get('duration_minutes', 0):.1f} minutes")
    print(f"Total Samples: {len(measurements)}")
    print(f"Peak Memory: {data.get('peak_memory_gb', 0):.1f}GB at {data.get('peak_timestamp')}")
    
    # Extract memory series
    ram_used_series = [m['system']['used_gb'] for m in measurements]
    ram_percent_series = [m['system']['percent'] for m in measurements]
    swap_percent_series = [m['system']['swap_percent'] for m in measurements]
    python_count_series = [m['processes']['num_python'] for m in measurements]
    ray_count_series = [m['processes']['num_ray'] for m in measurements]
    
    print("\n" + "-"*80)
    print("MEMORY STATISTICS")
    print("-"*80)
    
    print(f"\nRAM Usage:")
    print(f"  - Current:    {ram_used_series[-1]:.1f}GB")
    print(f"  - Average:    {sum(ram_used_series)/len(ram_used_series):.1f}GB")
    print(f"  - Peak:       {max(ram_used_series):.1f}GB")
    print(f"  - Min:        {min(ram_used_series):.1f}GB")
    print(f"  - Variance:   {max(ram_used_series) - min(ram_used_series):.1f}GB")
    
    print(f"\nRAM Percentage:")
    print(f"  - Current:    {ram_percent_series[-1]:.1f}%")
    print(f"  - Average:    {sum(ram_percent_series)/len(ram_percent_series):.1f}%")
    print(f"  - Peak:       {max(ram_percent_series):.1f}%")
    print(f"  - Critical (>90%): {'⚠ YES' if max(ram_percent_series) > 90 else '✓ No'}")
    
    print(f"\nSwap Usage:")
    print(f"  - Current:    {swap_percent_series[-1]:.1f}%")
    print(f"  - Average:    {sum(swap_percent_series)/len(swap_percent_series):.1f}%")
    print(f"  - Peak:       {max(swap_percent_series):.1f}%")
    print(f"  - Active (>10%): {'⚠ YES' if max(swap_percent_series) > 10 else '✓ No'}")
    
    print(f"\nProcess Counts:")
    print(f"  - Python processes (current): {python_count_series[-1]}")
    print(f"  - Python processes (peak):    {max(python_count_series)}")
    print(f"  - Ray workers (current):      {ray_count_series[-1]}")
    print(f"  - Ray workers (peak):         {max(ray_count_series)}")
    
    # Show top memory consumers
    if measurements:
        last_sample = measurements[-1]
        procs = last_sample['processes'].get('python_processes', [])
        
        if procs:
            print(f"\n" + "-"*80)
            print("TOP MEMORY CONSUMERS (Python processes)")
            print("-"*80)
            
            procs_sorted = sorted(procs, key=lambda x: x['memory_mb'], reverse=True)[:10]
            print(f"\n{'PID':<8} {'Memory':<12} {'Process':<50}")
            print("-" * 70)
            
            for p in procs_sorted:
                cmd_short = p['cmd'][:45]
                print(f"{p['pid']:<8} {p['memory_mb']:>8.1f}MB   {cmd_short:<45}")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    peak_percent = max(ram_percent_series) if ram_percent_series else 0
    peak_swap = max(swap_percent_series) if swap_percent_series else 0
    
    if peak_percent > 95:
        print("\n🔴 CRITICAL: Memory pressure > 95%")
        print("   └─ You're hitting RAM limits")
        print("   └─ SOLUTION: Upgrade RAM or reduce clients")
    elif peak_percent > 85:
        print("\n🟡 WARNING: Memory pressure 85-95%")
        print("   └─ Getting close to limits")
        print("   └─ Monitor swap usage closely")
    else:
        print("\n✅ Memory pressure is healthy (<85%)")
        print("   └─ Can handle current workload")
    
    if peak_swap > 50:
        print("\n🔴 CRITICAL: Swap > 50%")
        print("   └─ Severe swapping happening")
        print("   └─ Causing massive slowdowns")
        print("   └─ SOLUTION: Add more RAM or use fewer clients")
    elif peak_swap > 10:
        print("\n🟡 WARNING: Swap 10-50%")
        print("   └─ Some disk I/O happening")
        print("   └─ Consider reducing parallelism")
    else:
        print("\n✅ Swap usage minimal (<10%)")
        print("   └─ Good, staying in RAM")
    
    # Calculate memory per client
    if measurements:
        avg_ram = sum(ram_used_series) / len(ram_used_series)
        # Rough estimate: 16 parallel clients (from 8 vCPU / 0.5)
        ram_per_client = avg_ram / 16
        print(f"\nEstimated Memory per Parallel Client:")
        print(f"  {ram_per_client:.2f}GB per client")
        print(f"  ({ram_per_client*1024:.0f}MB per client)")

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'ram_measurements.json'
    analyze_ram_log(log_file)
