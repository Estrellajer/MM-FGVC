"""Generate task-dependent component accuracy heatmaps from HRP diagnostics JSONs."""
import json
import os
import glob
import numpy as np
import sys

# Try to use matplotlib with a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# --- Configuration ---
# Use C1 main results (seed42, qwen2_vl) for the heatmap
BASE_DIR = r"d:\Code_copy\MM-FGVC\swap\paper\outputs\20260325_095501_c1_main_results"
OUTPUT_DIR = r"d:\Code_copy\MM-FGVC\swap\paper\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select 6 representative tasks across categories
TASKS = {
    "cub_small":        {"label": "CUB-200\n(FGVC)", "category": "FGVC"},
    "flowers_small":    {"label": "Flowers-102\n(FGVC)", "category": "FGVC"},
    "eurosat_small":    {"label": "EuroSAT\n(Remote Sensing)", "category": "FGVC"},
    "vlguard":          {"label": "VLGuard\n(Safety)", "category": "Safety"},
    "naturalbench_vqa": {"label": "NaturalBench\n(VQA)", "category": "VQA"},
    "blink_relative_depth": {"label": "BLINK Depth\n(Reasoning)", "category": "BLINK"},
}

LEVELS = ["head", "attn", "mlp", "layer"]
LEVEL_LABELS = {"head": "Head", "attn": "Attn", "mlp": "MLP", "layer": "Layer"}
MODEL = "qwen2_vl"
SEED = "seed42"


def load_diagnostics(task_name):
    """Load a diagnostics JSON for the given task."""
    pattern = os.path.join(BASE_DIR, f"c1_main_results_{task_name}_{SEED}_rsev2_{MODEL}_*.diagnostics.json")
    files = glob.glob(pattern)
    if not files:
        print(f"  WARNING: No diagnostics found for {task_name} (pattern: {pattern})")
        return None
    with open(files[0], 'r') as f:
        return json.load(f)


def extract_heatmap_data(diag):
    """Extract a (4, L) matrix of val_accuracy from diagnostics."""
    num_layers = diag["num_layers"]
    matrix = np.full((4, num_layers), np.nan)
    
    for comp in diag["selected_components"]:
        level = comp["level"]
        layer = comp["layer_idx"]
        acc = comp.get("val_accuracy")
        if acc is not None and level in LEVELS:
            row = LEVELS.index(level)
            matrix[row, layer] = acc
    
    return matrix


def main():
    print("Generating heatmaps...")
    
    # Collect data
    task_data = {}
    for task_name, info in TASKS.items():
        print(f"  Loading {task_name}...")
        diag = load_diagnostics(task_name)
        if diag is not None:
            matrix = extract_heatmap_data(diag)
            task_data[task_name] = {
                "matrix": matrix,
                "label": info["label"],
                "category": info["category"],
                "num_layers": diag["num_layers"],
            }
    
    if not task_data:
        print("ERROR: No data loaded!")
        return
    
    n_tasks = len(task_data)
    print(f"  Loaded {n_tasks} tasks")
    
    # --- Create figure ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=150)
    axes = axes.flatten()
    
    # Custom colormap: dark blue (low) -> white (mid) -> dark red (high)
    cmap = LinearSegmentedColormap.from_list(
        "custom_rdbu",
        ["#2166AC", "#67A9CF", "#D1E5F0", "#FDDBC7", "#EF8A62", "#B2182B"],
        N=256
    )
    
    for idx, (task_name, data) in enumerate(task_data.items()):
        ax = axes[idx]
        matrix = data["matrix"]
        num_layers = data["num_layers"]
        
        # Plot heatmap
        im = ax.imshow(
            matrix, 
            aspect='auto', 
            cmap=cmap,
            interpolation='nearest',
            vmin=0.0,
            vmax=1.0,
        )
        
        # Labels
        ax.set_title(data["label"], fontsize=11, fontweight='bold', pad=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels([LEVEL_LABELS[l] for l in LEVELS], fontsize=9)
        
        # X-axis: show every 4th layer
        xtick_positions = list(range(0, num_layers, 4))
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([str(i) for i in xtick_positions], fontsize=8)
        ax.set_xlabel("Layer Index", fontsize=9)
        
        # Mark the best component with a star
        best_val = np.nanmax(matrix)
        best_pos = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        ax.plot(best_pos[1], best_pos[0], '*', color='gold', markersize=14, 
                markeredgecolor='black', markeredgewidth=0.8, zorder=10)
        ax.text(best_pos[1], best_pos[0] - 0.45, f'{best_val:.2f}', 
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Shared colorbar
    fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Component Val Accuracy', fontsize=10)
    
    fig.suptitle(
        'Task-Dependent Discriminative Power Across Decoder Layers and Representation Levels',
        fontsize=13, fontweight='bold', y=0.98
    )
    
    # Save SVG first (vector, no DPI issues)
    out_svg = os.path.join(OUTPUT_DIR, "f2_heatmap.svg")
    fig.savefig(out_svg, bbox_inches='tight', facecolor='white', format='svg')
    print(f"  Saved to {out_svg}")
    
    # Save PNG at lower DPI to avoid PIL issues
    out_path = os.path.join(OUTPUT_DIR, "f2_heatmap.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=100)
    print(f"  Saved to {out_path}")
    
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
