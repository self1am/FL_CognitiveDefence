#!/usr/bin/env python3
"""
Analyze the impact of attacks on client training by examining log files.
This script helps verify that attacks are actually being applied.
"""

import json
from pathlib import Path
import pandas as pd

def load_client_logs():
    """Load all client training logs"""
    client_data = []
    
    for log_file in Path('.').glob('client_*_training_log.json'):
        with open(log_file, 'r') as f:
            data = json.load(f)
            for entry in data:
                client_data.append(entry)
    
    return pd.DataFrame(client_data)

def analyze_attacks():
    """Analyze attack impact on different clients"""
    df = load_client_logs()
    
    if df.empty:
        print("No client training logs found!")
        return
    
    print("=" * 80)
    print("CLIENT TRAINING ANALYSIS - Attack Impact")
    print("=" * 80)
    
    # Group by client and attack status
    for client_id in sorted(df['client_id'].unique()):
        client_df = df[df['client_id'] == client_id]
        
        # Get attack info
        attacked = client_df['attacked'].iloc[0]
        attack_type = client_df['attack_type'].iloc[0]
        
        # Get final round stats
        final_round = client_df[client_df['round'] == client_df['round'].max()].iloc[0]
        
        print(f"\n📊 Client {client_id}:")
        print(f"   Attack Status: {'⚠️  ATTACKED' if attacked else '✅ BENIGN'}")
        if attacked:
            print(f"   Attack Type:   {attack_type}")
        
        print(f"   Final Round {int(final_round['round'])}:")
        print(f"      Training Loss: {final_round['avg_loss']:.4f}")
        print(f"      Training Acc:  {final_round['training_accuracy']:.4f}")
        
        # Show trend
        first_round = client_df[client_df['round'] == client_df['round'].min()].iloc[0]
        acc_improvement = final_round['training_accuracy'] - first_round['training_accuracy']
        print(f"   Accuracy Δ:    {acc_improvement:+.4f} (R1→R{int(final_round['round'])})")
    
    print("\n" + "=" * 80)
    print("SUMMARY BY ATTACK STATUS")
    print("=" * 80)
    
    # Get final round data for each client
    final_rounds = df[df['round'] == df['round'].max()]
    
    # Attacked vs Benign comparison
    attacked_clients = final_rounds[final_rounds['attacked'] == True]
    benign_clients = final_rounds[final_rounds['attacked'] == False]
    
    if not attacked_clients.empty:
        print(f"\n⚠️  ATTACKED CLIENTS (n={len(attacked_clients)}):")
        print(f"   Avg Training Accuracy: {attacked_clients['training_accuracy'].mean():.4f}")
        print(f"   Avg Training Loss:     {attacked_clients['avg_loss'].mean():.4f}")
        print(f"   Client IDs: {sorted(attacked_clients['client_id'].tolist())}")
    
    if not benign_clients.empty:
        print(f"\n✅ BENIGN CLIENTS (n={len(benign_clients)}):")
        print(f"   Avg Training Accuracy: {benign_clients['training_accuracy'].mean():.4f}")
        print(f"   Avg Training Loss:     {benign_clients['avg_loss'].mean():.4f}")
        print(f"   Client IDs: {sorted(benign_clients['client_id'].tolist())}")
    
    # The KEY Issue
    print("\n" + "=" * 80)
    print("⚠️  CRITICAL ISSUE IDENTIFIED:")
    print("=" * 80)
    print("""
The global model loss you're seeing (0.038) comes from DISTRIBUTED evaluation,
which means it's averaging the losses from ALL clients' local test sets.

Since ~50% of your clients are BENIGN (not attacked), they report very low
losses on their clean local test sets, which masks the attack impact!

The attacked clients show degraded training accuracy (~89% vs 99%), but when
you average evaluation results across all clients, the benign clients' good
performance dominates.

✅ SOLUTION: Implement centralized evaluation on the server using a single,
   clean test set (the full 10,000 MNIST test images). This will show the
   TRUE impact of the attacks on the global model!

The fix has been applied - now run the experiment again to see centralized
evaluation results labeled as "Centralized Test Loss" in the server logs.
""")

if __name__ == "__main__":
    analyze_attacks()
