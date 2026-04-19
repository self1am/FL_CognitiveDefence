"""
Visualization script for FL Cognitive Defence experiments
Generates charts comparing baseline and attack-only scenarios
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_client_logs(experiment_path):
    """Load all client training logs from an experiment directory"""
    experiment_dir = Path(experiment_path)
    client_logs = {}
    
    for log_file in experiment_dir.glob("client_*_training_log.json"):
        with open(log_file, 'r') as f:
            data = json.load(f)
            if data:
                client_id = data[0]['client_id']
                client_logs[client_id] = data
    
    return client_logs

def load_global_metrics(experiment_path):
    """Parse global metrics from final_log.txt"""
    log_file = Path(experiment_path) / "final_log.txt"
    rounds = []
    losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'round' in line and ':' in line and 'round(s)' not in line:
                parts = line.split('round')[1].split(':')
                if len(parts) == 2:
                    round_num = int(parts[0].strip())
                    loss = float(parts[1].strip())
                    rounds.append(round_num)
                    losses.append(loss)
    
    return rounds, losses

def plot_accuracy_comparison(baseline_logs, attack_logs, output_dir):
    """Compare training accuracy across experiments"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline accuracy
    for client_id, logs in sorted(baseline_logs.items()):
        rounds = [entry['round'] for entry in logs]
        accuracies = [entry['training_accuracy'] * 100 for entry in logs]
        ax1.plot(rounds, accuracies, marker='o', label=f'Client {client_id}', linewidth=2)
    
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline: Training Accuracy Over Rounds', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([80, 100])
    
    # Attack scenario accuracy
    for client_id, logs in sorted(attack_logs.items()):
        rounds = [entry['round'] for entry in logs]
        accuracies = [entry['training_accuracy'] * 100 for entry in logs]
        is_attacked = logs[0].get('attacked', False)
        linestyle = '--' if is_attacked else '-'
        marker = 'x' if is_attacked else 'o'
        label = f'Client {client_id} (attacked)' if is_attacked else f'Client {client_id}'
        ax2.plot(rounds, accuracies, marker=marker, linestyle=linestyle, 
                label=label, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Attack Scenario: Training Accuracy Over Rounds', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([80, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: accuracy_comparison.png")
    plt.close()

def plot_loss_comparison(baseline_logs, attack_logs, output_dir):
    """Compare training loss across experiments"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline loss
    for client_id, logs in sorted(baseline_logs.items()):
        rounds = [entry['round'] for entry in logs]
        losses = [entry['avg_loss'] for entry in logs]
        ax1.plot(rounds, losses, marker='o', label=f'Client {client_id}', linewidth=2)
    
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline: Training Loss Over Rounds', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Attack scenario loss
    for client_id, logs in sorted(attack_logs.items()):
        rounds = [entry['round'] for entry in logs]
        losses = [entry['avg_loss'] for entry in logs]
        is_attacked = logs[0].get('attacked', False)
        linestyle = '--' if is_attacked else '-'
        marker = 'x' if is_attacked else 'o'
        label = f'Client {client_id} (attacked)' if is_attacked else f'Client {client_id}'
        ax2.plot(rounds, losses, marker=marker, linestyle=linestyle, 
                label=label, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Attack Scenario: Training Loss Over Rounds', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: loss_comparison.png")
    plt.close()

def plot_global_loss_comparison(baseline_path, attack_path, output_dir):
    """Compare global model loss between scenarios"""
    baseline_rounds, baseline_losses = load_global_metrics(baseline_path)
    attack_rounds, attack_losses = load_global_metrics(attack_path)
    
    plt.figure(figsize=(12, 7))
    plt.plot(baseline_rounds, baseline_losses, marker='o', linewidth=3, 
            markersize=10, label='Baseline (No Attack)', color='#2ecc71')
    plt.plot(attack_rounds, attack_losses, marker='s', linewidth=3, 
            markersize=10, label='Attack Scenario (Label Flip)', color='#e74c3c')
    
    plt.xlabel('Round', fontsize=14, fontweight='bold')
    plt.ylabel('Global Model Loss', fontsize=14, fontweight='bold')
    plt.title('Global Model Convergence: Baseline vs Attack Scenario', 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add annotations for final values
    plt.annotate(f'Final: {baseline_losses[-1]:.4f}', 
                xy=(baseline_rounds[-1], baseline_losses[-1]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#2ecc71', alpha=0.7),
                fontsize=10, fontweight='bold')
    plt.annotate(f'Final: {attack_losses[-1]:.4f}', 
                xy=(attack_rounds[-1], attack_losses[-1]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#e74c3c', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'global_loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: global_loss_comparison.png")
    plt.close()

def plot_final_accuracy_bar_chart(baseline_logs, attack_logs, output_dir):
    """Bar chart comparing final accuracy of all clients"""
    baseline_final = {}
    attack_final = {}
    
    for client_id, logs in baseline_logs.items():
        baseline_final[client_id] = logs[-1]['training_accuracy'] * 100
    
    for client_id, logs in attack_logs.items():
        attack_final[client_id] = logs[-1]['training_accuracy'] * 100
    
    # Combine all client IDs
    all_clients = sorted(set(list(baseline_final.keys()) + list(attack_final.keys())))
    
    # Identify attacked clients
    attacked_clients = set()
    for client_id, logs in attack_logs.items():
        if logs[0].get('attacked', False):
            attacked_clients.add(client_id)
    
    x = np.arange(len(all_clients))
    width = 0.35
    
    baseline_vals = [baseline_final.get(c, 0) for c in all_clients]
    attack_vals = [attack_final.get(c, 0) for c in all_clients]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db')
    bars2 = ax.bar(x + width/2, attack_vals, width, label='Attack Scenario', color='#e67e22')
    
    # Highlight attacked clients
    for i, client_id in enumerate(all_clients):
        if client_id in attacked_clients:
            ax.bar(i + width/2, attack_vals[i], width, color='#c0392b', 
                  edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Client ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Final Training Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Final Training Accuracy by Client: Baseline vs Attack Scenario', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_clients)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([85, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    
    # Add legend for attacked clients
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Baseline'),
        Patch(facecolor='#e67e22', label='Attack Scenario (Benign)'),
        Patch(facecolor='#c0392b', edgecolor='black', linewidth=2, 
              label='Attack Scenario (Malicious)')
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: final_accuracy_comparison.png")
    plt.close()

def plot_attack_impact_heatmap(baseline_logs, attack_logs, output_dir):
    """Heatmap showing accuracy degradation per client per round"""
    # Find common clients
    common_clients = sorted(set(baseline_logs.keys()) & set(attack_logs.keys()))
    
    if not common_clients:
        print("⚠ No common clients found for heatmap")
        return
    
    # Get max rounds
    max_rounds = max(
        max(len(logs) for logs in baseline_logs.values()),
        max(len(logs) for logs in attack_logs.values())
    )
    
    # Calculate accuracy difference matrix
    diff_matrix = []
    for client_id in common_clients:
        baseline = baseline_logs[client_id]
        attack = attack_logs[client_id]
        
        row = []
        for round_idx in range(min(len(baseline), len(attack))):
            baseline_acc = baseline[round_idx]['training_accuracy'] * 100
            attack_acc = attack[round_idx]['training_accuracy'] * 100
            diff = baseline_acc - attack_acc
            row.append(diff)
        diff_matrix.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(diff_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=15)
    
    # Set ticks
    ax.set_xticks(np.arange(len(diff_matrix[0])))
    ax.set_yticks(np.arange(len(common_clients)))
    ax.set_xticklabels([f'R{i+1}' for i in range(len(diff_matrix[0]))])
    ax.set_yticklabels([f'Client {c}' for c in common_clients])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy Degradation (%)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(common_clients)):
        for j in range(len(diff_matrix[0])):
            text = ax.text(j, i, f'{diff_matrix[i][j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_xlabel('Training Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Client ID', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Degradation Heatmap: Baseline - Attack Scenario (%)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_degradation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: accuracy_degradation_heatmap.png")
    plt.close()

def plot_convergence_rate(baseline_logs, attack_logs, output_dir):
    """Plot convergence rate comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate average accuracy per round
    def get_avg_accuracy_per_round(logs_dict):
        rounds_data = {}
        for client_id, logs in logs_dict.items():
            for entry in logs:
                round_num = entry['round']
                acc = entry['training_accuracy'] * 100
                if round_num not in rounds_data:
                    rounds_data[round_num] = []
                rounds_data[round_num].append(acc)
        
        rounds = sorted(rounds_data.keys())
        avg_accs = [np.mean(rounds_data[r]) for r in rounds]
        std_accs = [np.std(rounds_data[r]) for r in rounds]
        return rounds, avg_accs, std_accs
    
    baseline_rounds, baseline_avg, baseline_std = get_avg_accuracy_per_round(baseline_logs)
    attack_rounds, attack_avg, attack_std = get_avg_accuracy_per_round(attack_logs)
    
    # Plot with confidence intervals
    ax.plot(baseline_rounds, baseline_avg, marker='o', linewidth=3, 
           markersize=10, label='Baseline (Mean)', color='#27ae60')
    ax.fill_between(baseline_rounds, 
                     np.array(baseline_avg) - np.array(baseline_std),
                     np.array(baseline_avg) + np.array(baseline_std),
                     alpha=0.2, color='#27ae60', label='Baseline (±1 std)')
    
    ax.plot(attack_rounds, attack_avg, marker='s', linewidth=3, 
           markersize=10, label='Attack Scenario (Mean)', color='#c0392b')
    ax.fill_between(attack_rounds, 
                     np.array(attack_avg) - np.array(attack_std),
                     np.array(attack_avg) + np.array(attack_std),
                     alpha=0.2, color='#c0392b', label='Attack Scenario (±1 std)')
    
    ax.set_xlabel('Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Training Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Convergence Rate: Mean Accuracy Across All Clients', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([85, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_rate.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: convergence_rate.png")
    plt.close()

def generate_summary_statistics(baseline_logs, attack_logs, output_dir):
    """Generate and save summary statistics"""
    summary = []
    summary.append("=" * 80)
    summary.append("EXPERIMENT SUMMARY STATISTICS")
    summary.append("=" * 80)
    summary.append("")
    
    # Baseline stats
    summary.append("BASELINE EXPERIMENT:")
    summary.append("-" * 40)
    baseline_final_accs = [logs[-1]['training_accuracy'] * 100 
                          for logs in baseline_logs.values()]
    summary.append(f"  Number of clients: {len(baseline_logs)}")
    summary.append(f"  Average final accuracy: {np.mean(baseline_final_accs):.2f}%")
    summary.append(f"  Std dev final accuracy: {np.std(baseline_final_accs):.2f}%")
    summary.append(f"  Min final accuracy: {np.min(baseline_final_accs):.2f}%")
    summary.append(f"  Max final accuracy: {np.max(baseline_final_accs):.2f}%")
    summary.append("")
    
    # Attack stats
    summary.append("ATTACK SCENARIO:")
    summary.append("-" * 40)
    attack_final_accs = [logs[-1]['training_accuracy'] * 100 
                        for logs in attack_logs.values()]
    attacked_clients = [logs[-1]['training_accuracy'] * 100 
                       for logs in attack_logs.values() 
                       if logs[0].get('attacked', False)]
    benign_clients = [logs[-1]['training_accuracy'] * 100 
                     for logs in attack_logs.values() 
                     if not logs[0].get('attacked', False)]
    
    summary.append(f"  Total clients: {len(attack_logs)}")
    summary.append(f"  Attacked clients: {len(attacked_clients)}")
    summary.append(f"  Benign clients: {len(benign_clients)}")
    summary.append(f"  Overall average final accuracy: {np.mean(attack_final_accs):.2f}%")
    summary.append("")
    
    if attacked_clients:
        summary.append(f"  Attacked clients final accuracy: {np.mean(attacked_clients):.2f}%")
        summary.append(f"  Attacked clients std dev: {np.std(attacked_clients):.2f}%")
    
    if benign_clients:
        summary.append(f"  Benign clients final accuracy: {np.mean(benign_clients):.2f}%")
        summary.append(f"  Benign clients std dev: {np.std(benign_clients):.2f}%")
    summary.append("")
    
    # Comparison
    summary.append("IMPACT ANALYSIS:")
    summary.append("-" * 40)
    accuracy_drop = np.mean(baseline_final_accs) - np.mean(attack_final_accs)
    summary.append(f"  Average accuracy drop: {accuracy_drop:.2f}%")
    
    if attacked_clients:
        attacked_drop = np.mean(baseline_final_accs) - np.mean(attacked_clients)
        summary.append(f"  Attacked clients accuracy drop: {attacked_drop:.2f}%")
    
    summary.append("=" * 80)
    
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write(summary_text)
    print(f"\n✓ Saved: summary_statistics.txt")

def main():
    # Define paths
    base_path = Path(__file__).parent / "results"
    baseline_path = base_path / "baseline"
    attack_path = base_path / "attack-only"
    output_dir = base_path / "visualizations"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("FL COGNITIVE DEFENCE - EXPERIMENT VISUALIZATION")
    print("=" * 80)
    print()
    
    # Load data
    print("📂 Loading experiment data...")
    baseline_logs = load_client_logs(baseline_path)
    attack_logs = load_client_logs(attack_path)
    print(f"   Baseline: {len(baseline_logs)} clients")
    print(f"   Attack Scenario: {len(attack_logs)} clients")
    print()
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    print()
    
    plot_accuracy_comparison(baseline_logs, attack_logs, output_dir)
    plot_loss_comparison(baseline_logs, attack_logs, output_dir)
    plot_global_loss_comparison(baseline_path, attack_path, output_dir)
    plot_final_accuracy_bar_chart(baseline_logs, attack_logs, output_dir)
    plot_attack_impact_heatmap(baseline_logs, attack_logs, output_dir)
    plot_convergence_rate(baseline_logs, attack_logs, output_dir)
    
    print()
    print("📈 Generating summary statistics...")
    generate_summary_statistics(baseline_logs, attack_logs, output_dir)
    
    print()
    print("=" * 80)
    print(f"✅ All visualizations saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
