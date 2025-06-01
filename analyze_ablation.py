import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(base_dir="../experiments/ablation"):
    """
    Load all ablation study results into a pandas DataFrame
    """
    results = []
    datasets = ['360', 'tandt']
    methods = ['baseline', 'full', 'no_culling', 'no_patch_compare', 'no_ds', 'no_lap', 'random_cameras']
    
    # Map method names to more readable labels
    method_labels = {
        'baseline': 'Baseline (No Aug)',
        'full': 'Full Method',
        'no_culling': 'No Visibility Culling',
        'no_patch_compare': 'No Patch Compare',
        'no_ds': 'No Depth Smoothness',
        'no_lap': 'No Laplacian Loss',
        'random_cameras': 'Random Camera Order'
    }
    
    for dataset in datasets:
        dataset_dir = Path(base_dir) / dataset
        if not dataset_dir.exists():
            continue
            
        for scene_dir in dataset_dir.iterdir():
            if not scene_dir.is_dir():
                continue
                
            # Extract scene name and method from directory name
            parts = scene_dir.name.split('_')
            scene = parts[0]
            method = '_'.join(parts[1:])
            
            if method not in methods:
                continue
                
            # Load results from JSON file
            result_file = scene_dir / "results.json"
            if not result_file.exists():
                continue
                
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract metrics (first method in results)
                metrics = list(data.values())[0]
                results.append({
                    'Dataset': dataset,
                    'Scene': scene,
                    'Method': method,
                    'Method_Label': method_labels.get(method, method),
                    'PSNR': metrics.get('PSNR', 0),
                    'SSIM': metrics.get('SSIM', 0),
                    'LPIPS': metrics.get('LPIPS', 0)
                })
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=['Dataset', 'Scene', 'Method', 'Method_Label', 'PSNR', 'SSIM', 'LPIPS'])

def analyze_results(df):
    """
    Print statistical analysis of the results
    """
    if df.empty:
        print("No results to analyze")
        return
    
    # Group by method and calculate mean metrics
    method_summary = df.groupby('Method_Label')[['PSNR', 'SSIM', 'LPIPS']].mean()
    
    # Calculate improvement over baseline
    baseline_metrics = method_summary.loc['Baseline (No Aug)']
    improvements = method_summary.copy()
    
    # For PSNR and SSIM, higher is better
    improvements['PSNR'] = method_summary['PSNR'] - baseline_metrics['PSNR']
    improvements['SSIM'] = method_summary['SSIM'] - baseline_metrics['SSIM']
    
    # For LPIPS, lower is better
    improvements['LPIPS'] = baseline_metrics['LPIPS'] - method_summary['LPIPS']
    
    print("Mean metrics by method:")
    print(method_summary)
    print("\nImprovement over baseline:")
    print(improvements)
    
    # Per-dataset analysis
    print("\nPer-dataset metrics:")
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        dataset_summary = dataset_df.groupby('Method_Label')[['PSNR', 'SSIM', 'LPIPS']].mean()
        print(f"\n{dataset} dataset:")
        print(dataset_summary)

def plot_metrics(df, output_dir="../experiments/ablation/figures"):
    """
    Generate visualizations of the ablation study results
    """
    if df.empty:
        print("No results to plot")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Plot mean metrics by method
    metrics = ['PSNR', 'SSIM', 'LPIPS']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # For barplots
        summary = df.groupby('Method_Label')[metric].mean().reset_index()
        # Sort methods with baseline first, then full method, then the rest
        method_order = ['Baseline (No Aug)', 'Full Method'] + [m for m in summary['Method_Label'].unique() 
                                                             if m not in ['Baseline (No Aug)', 'Full Method']]
        summary['Method_Label'] = pd.Categorical(summary['Method_Label'], categories=method_order, ordered=True)
        summary = summary.sort_values('Method_Label')
        
        # Create bar plot
        ax = sns.barplot(x='Method_Label', y=metric, data=summary)
        
        # Add value labels on bars
        for i, v in enumerate(summary[metric]):
            ax.text(i, v + (0.01 * max(summary[metric])), f"{v:.3f}", ha='center')
        
        # Adjust layout
        plt.title(f"Mean {metric} by Method")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{metric}_by_method.png"), dpi=300)
        plt.close()
    
    # Plot per-scene comparison
    for metric in metrics:
        # Group by scene and method
        pivot_df = df.pivot(index='Scene', columns='Method_Label', values=metric)
        
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"{metric} Comparison by Scene and Method")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_by_scene_method.png"), dpi=300)
        plt.close()
    
    # Plot radar chart for comparing methods
    method_summary = df.groupby('Method_Label')[metrics].mean()
    
    # Normalize metrics for radar chart (0-1 scale)
    normalized = method_summary.copy()
    for metric in metrics:
        if metric == 'LPIPS':  # Lower is better for LPIPS
            normalized[metric] = 1 - (method_summary[metric] - method_summary[metric].min()) / (method_summary[metric].max() - method_summary[metric].min())
        else:  # Higher is better for PSNR, SSIM
            normalized[metric] = (method_summary[metric] - method_summary[metric].min()) / (method_summary[metric].max() - method_summary[metric].min())
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each metric
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each method
    for method in normalized.index:
        values = normalized.loc[method].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.1)
    
    # Set the labels
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 1.1)
    plt.title("Normalized Performance Comparison")
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart_comparison.png"), dpi=300)
    plt.close()
    
    # Plot boxplots to show distribution across scenes
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Method_Label', y=metric, data=df, order=method_order)
        plt.title(f"Distribution of {metric} Across Scenes")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_boxplot.png"), dpi=300)
        plt.close()

def generate_report(df, output_dir="../experiments/ablation"):
    """
    Generate a markdown report summarizing the ablation study results
    """
    if df.empty:
        print("No results to report")
        return
    
    # Create summary tables
    method_summary = df.groupby('Method_Label')[['PSNR', 'SSIM', 'LPIPS']].mean()
    
    # Calculate improvements
    baseline_metrics = method_summary.loc['Baseline (No Aug)']
    improvements = method_summary.copy()
    improvements['PSNR'] = method_summary['PSNR'] - baseline_metrics['PSNR']
    improvements['SSIM'] = method_summary['SSIM'] - baseline_metrics['SSIM']
    improvements['LPIPS'] = baseline_metrics['LPIPS'] - method_summary['LPIPS']
    
    # Generate markdown report
    report = "# Ablation Study Results\n\n"
    
    report += "## Mean Metrics by Method\n\n"
    report += method_summary.to_markdown() + "\n\n"
    
    report += "## Improvements Over Baseline\n\n"
    report += improvements.to_markdown() + "\n\n"
    
    report += "## Per-Dataset Results\n\n"
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        dataset_summary = dataset_df.groupby('Method_Label')[['PSNR', 'SSIM', 'LPIPS']].mean()
        report += f"### {dataset} Dataset\n\n"
        report += dataset_summary.to_markdown() + "\n\n"
    
    report += "## Key Findings\n\n"
    report += "- The full method shows an improvement of {:.3f} dB in PSNR over the baseline.\n".format(
        improvements.loc['Full Method', 'PSNR'])
    report += "- Removing visibility culling results in a performance drop of {:.3f} dB in PSNR compared to the full method.\n".format(
        method_summary.loc['Full Method', 'PSNR'] - method_summary.loc['No Visibility Culling', 'PSNR'])
    
    # Add more key findings based on the data
    
    # Save the report
    with open(os.path.join(output_dir, "ablation_report.md"), 'w') as f:
        f.write(report)
    
    print(f"Report saved to {os.path.join(output_dir, 'ablation_report.md')}")

def main():
    # Load results
    df = load_results()
    
    if df.empty:
        print("No ablation study results found. Run the ablation_study.sh script first.")
        return
    
    # Analyze results
    analyze_results(df)
    
    # Generate visualizations
    plot_metrics(df)
    
    # Generate report
    generate_report(df)

if __name__ == "__main__":
    main() 