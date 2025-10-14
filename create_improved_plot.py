#!/usr/bin/env python3
"""
Create an improved plot that clearly shows seed numbers and rankings.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def create_improved_plot(json_file_path, output_dir):
    """Create an improved plot with clear seed labels"""
    
    # Load the data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    seed_data = data['seed_data']
    summary_stats = data['summary_stats']
    
    # Sort by mean loss
    sorted_data = sorted(seed_data, key=lambda x: x['mean_loss'])
    
    # Create figure with better layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Seed Analysis Results - {len(seed_data)} Seeds', fontsize=18, fontweight='bold')
    
    # Extract data
    mean_losses = [data['mean_loss'] for data in sorted_data]
    std_losses = [data['std_loss'] for data in sorted_data]
    seeds = [data['seed'] for data in sorted_data]
    
    # Plot 1: Bar chart of all seeds with seed numbers
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(seeds)), mean_losses, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Seed Rank (Best to Worst)')
    ax1.set_ylabel('Mean Training Loss')
    ax1.set_title('All Seeds Ranked by Mean Loss')
    ax1.grid(True, alpha=0.3)
    
    # Color the best seeds differently
    for i, (bar, seed) in enumerate(zip(bars, seeds)):
        if i < 3:  # Top 3 seeds
            bar.set_color('gold' if i == 0 else 'silver' if i == 1 else '#CD7F32')  # Gold, Silver, Bronze
        elif i < len(seeds) // 2:  # Top half
            bar.set_color('lightgreen')
        else:  # Bottom half
            bar.set_color('lightcoral')
    
    # Add seed numbers as text on bars
    for i, (bar, seed) in enumerate(zip(bars, seeds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Seed {seed}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Plot 2: Scatter plot with seed numbers
    ax2 = axes[0, 1]
    scatter = ax2.scatter(mean_losses, std_losses, alpha=0.7, s=100)
    ax2.set_xlabel('Mean Training Loss')
    ax2.set_ylabel('Std Training Loss')
    ax2.set_title('Mean vs Standard Deviation (with Seed Numbers)')
    ax2.grid(True, alpha=0.3)
    
    # Add seed numbers as annotations
    for i, (mean_loss, std_loss, seed) in enumerate(zip(mean_losses, std_losses, seeds)):
        ax2.annotate(f'Seed {seed}', (mean_loss, std_loss), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Color the best seeds
    colors = ['gold' if i == 0 else 'silver' if i == 1 else '#CD7F32' if i == 2 
              else 'lightgreen' if i < len(seeds)//2 else 'lightcoral' 
              for i in range(len(seeds))]
    scatter.set_color(colors)
    
    # Plot 3: Top 10 seeds detailed view
    ax3 = axes[1, 0]
    top_10 = sorted_data[:10]
    top_10_seeds = [data['seed'] for data in top_10]
    top_10_means = [data['mean_loss'] for data in top_10]
    
    bars = ax3.bar(range(len(top_10_seeds)), top_10_means, 
                   color=['gold', 'silver', '#CD7F32'] + ['lightgreen'] * 7)
    ax3.set_xlabel('Seed Number')
    ax3.set_ylabel('Mean Training Loss')
    ax3.set_title('Top 10 Seeds (Best to Worst)')
    ax3.set_xticks(range(len(top_10_seeds)))
    ax3.set_xticklabels([f'Seed {s}' for s in top_10_seeds], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, (bar, mean_loss) in enumerate(zip(bars, top_10_means)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{mean_loss:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    summary_text = f"""
🏆 BEST SEEDS RESULTS 🏆

🥇 BEST: Seed {sorted_data[0]['seed']}
   Mean Loss: {sorted_data[0]['mean_loss']:.6f}
   Std Loss: {sorted_data[0]['std_loss']:.6f}
   Samples: {sorted_data[0]['num_samples']:,}

🥈 2nd: Seed {sorted_data[1]['seed']}
   Mean Loss: {sorted_data[1]['mean_loss']:.6f}
   Std Loss: {sorted_data[1]['std_loss']:.6f}
   Samples: {sorted_data[1]['num_samples']:,}

🥉 3rd: Seed {sorted_data[2]['seed']}
   Mean Loss: {sorted_data[2]['mean_loss']:.6f}
   Std Loss: {sorted_data[2]['std_loss']:.6f}
   Samples: {sorted_data[2]['num_samples']:,}

📊 OVERALL STATS:
   Total Seeds: {len(seed_data)}
   Best Mean: {summary_stats['min_mean_loss']:.6f}
   Worst Mean: {summary_stats['max_mean_loss']:.6f}
   Overall Mean: {summary_stats['overall_mean_loss']:.6f}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'improved_seed_analysis_plot.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Improved plot saved to: {plot_path}")
    return plot_path

if __name__ == "__main__":
    # Use the existing results
    json_file = "test_results/seed_analysis_final_20251006_121010.json"
    output_dir = "test_results"
    
    if os.path.exists(json_file):
        plot_path = create_improved_plot(json_file, output_dir)
        print(f"\n🎯 The improved plot shows:")
        print(f"   - Seed numbers clearly labeled")
        print(f"   - Best seeds highlighted in gold/silver/bronze")
        print(f"   - Top 10 seeds with exact values")
        print(f"   - Summary statistics table")
        print(f"\n📁 View the plot: {plot_path}")
    else:
        print(f"❌ Results file not found: {json_file}")
