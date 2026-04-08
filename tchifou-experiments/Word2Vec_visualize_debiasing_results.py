import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
INPUT_CSV = os.path.join(OUTPUT_DIR, 'debiasing_metrics_results.csv')
PLOT_OUTPUT = os.path.join(OUTPUT_DIR, 'debiasing_analysis_plot.png')

def generate_plots():
    # Check if the results file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Results file not found at {INPUT_CSV}")
        print("Please run the analysis script first.")
        return

    # Load data
    df = pd.read_csv(INPUT_CSV)

    # Set up the figure with four subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact of Progressive PCA Removal on Gender Bias & Model Integrity', fontsize=16)

    # 1. Direct Bias (Lower is better)
    axes[0, 0].plot(df['pcs_removed'], df['direct_bias'], marker='o', color='blue')
    axes[0, 0].set_title('Direct Bias (Explicit)')
    axes[0, 0].set_ylabel('Mean Cosine Similarity')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # 2. WEAT Score (Lower is better)
    axes[0, 1].plot(df['pcs_removed'], df['weat_score'], marker='s', color='green')
    axes[0, 1].set_title('WEAT Effect Size (Implicit)')
    axes[0, 1].set_ylabel('Standardized Effect Size')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # 3. Mean Vector Displacement (MVD - Lower is better)
    axes[1, 0].plot(df['pcs_removed'], df['mvd'], marker='^', color='red')
    axes[1, 0].set_title('Mean Vector Displacement (Distortion)')
    axes[1, 0].set_xlabel('Number of PCs Removed')
    axes[1, 0].set_ylabel('Avg. Euclidean Distance')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    # 4. Semantic Geometry (Higher is better)
    axes[1, 1].plot(df['pcs_removed'], df['semantic_geometry'], marker='d', color='purple')
    axes[1, 1].set_title('Semantic Geometry (Accuracy)')
    axes[1, 1].set_xlabel('Number of PCs Removed')
    axes[1, 1].set_ylabel('Spearman Correlation')
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot to the output folder
    plt.savefig(PLOT_OUTPUT)
    print(f"Visualization saved to: {PLOT_OUTPUT}")
    plt.show()

if __name__ == "__main__":
    generate_plots()