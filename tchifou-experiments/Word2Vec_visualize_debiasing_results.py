import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
INPUT_CSV = os.path.join(OUTPUT_DIR, 'debiasing_metrics_results.csv')
PLOT_OUTPUT_DASHBOARD = os.path.join(OUTPUT_DIR, 'debiasing_dashboard.png')
PLOT_OUTPUT_COMBINED = os.path.join(OUTPUT_DIR, 'normalized_comparison_plot.png')

def min_max_normalize(series):
    """Scales a series to a range between 0 and 1."""
    if series.max() == series.min():
        return series * 0  # Avoid division by zero
    return (series - series.min()) / (series.max() - series.min())

def generate_plots():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Results file not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    # --- FIGURE 1: THE 4-PANEL DASHBOARD ---
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Individual Metric Trends per PCA Removal', fontsize=16)

    axes[0, 0].plot(df['pcs_removed'], df['direct_bias'], marker='o', color='blue')
    axes[0, 0].set_title('Direct Bias (Lower is Better)')
    axes[0, 1].plot(df['pcs_removed'], df['weat_score'], marker='s', color='green')
    axes[0, 1].set_title('WEAT Effect Size (Lower is Better)')
    axes[1, 0].plot(df['pcs_removed'], df['mvd'], marker='^', color='red')
    axes[1, 0].set_title('Mean Vector Displacement (Lower is Better)')
    axes[1, 1].plot(df['pcs_removed'], df['semantic_geometry'], marker='d', color='purple')
    axes[1, 1].set_title('Semantic Geometry (Higher is Better)')

    for ax in axes.flat:
        ax.set_xlabel('Number of PCs Removed')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(PLOT_OUTPUT_DASHBOARD)

    # --- FIGURE 2: THE NORMALIZED COMBINED PLOT ---
    plt.figure(figsize=(10, 6))
    
    # Calculate normalized values
    db_norm = min_max_normalize(df['direct_bias'])
    weat_norm = min_max_normalize(df['weat_score'])
    mvd_norm = min_max_normalize(df['mvd'])
    sem_norm = min_max_normalize(df['semantic_geometry'])

    # Plot all on the same axis
    plt.plot(df['pcs_removed'], db_norm, label='Direct Bias (Norm)', marker='o', linewidth=2)
    plt.plot(df['pcs_removed'], weat_norm, label='WEAT Score (Norm)', marker='s', linewidth=2)
    plt.plot(df['pcs_removed'], mvd_norm, label='Distortion / MVD (Norm)', marker='^', linewidth=2)
    plt.plot(df['pcs_removed'], sem_norm, label='Semantic Geometry (Norm)', marker='d', linewidth=2)

    plt.title('Combined Normalized Metric Comparison', fontsize=14)
    plt.xlabel('Number of Principal Components Removed')
    plt.ylabel('Normalized Scale (0 to 1)')
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)  
    
    plt.savefig(PLOT_OUTPUT_COMBINED)
    print(f"Individual Dashboard saved to: {PLOT_OUTPUT_DASHBOARD}")
    print(f"Normalized Comparison saved to: {PLOT_OUTPUT_COMBINED}")
    plt.show()

if __name__ == "__main__":
    generate_plots()