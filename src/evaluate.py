import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import sys

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


def load_wandb_config() -> Dict[str, str]:
    """Load WandB config from config/config.yaml."""
    from omegaconf import OmegaConf
    
    config_path = Path("config/config.yaml")
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        return dict(cfg.wandb)
    return {"entity": "gengaru617-personal", "project": "2025-11-19"}


def retrieve_run_data(run_id: str, wandb_config: Dict[str, str]) -> Dict[str, Any]:
    """Retrieve comprehensive metrics from WandB for a single run."""
    
    logger.info(f"Retrieving data for run: {run_id}")
    
    api = wandb.Api()
    run_path = f"{wandb_config['entity']}/{wandb_config['project']}/{run_id}"
    
    try:
        run = api.run(run_path)
    except Exception as e:
        logger.error(f"Failed to retrieve run {run_id}: {e}")
        return {}
    
    # Validate run completed
    if run.state != "finished":
        logger.warning(f"Run {run_id} has state '{run.state}', not 'finished'")
    
    # Get history (time-series metrics)
    history_df = run.history()
    
    # Validate history not empty
    if history_df is None or len(history_df) == 0:
        logger.warning(f"Run {run_id} has empty history")
    
    # Get summary (final metrics)
    summary = run.summary._json_dict
    
    # Get config
    config = dict(run.config)
    
    return {
        "run_id": run_id,
        "history": history_df,
        "summary": summary,
        "config": config,
        "status": run.state,
    }


def export_run_metrics(run_data: Dict[str, Any], output_dir: Path) -> Path:
    """Export run-specific metrics and time-series data to JSON."""
    
    run_id = run_data["run_id"]
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract key metrics from summary
    metrics = {
        "run_id": run_id,
        "status": run_data["status"],
        "primary_metrics": {
            "test_accuracy_at_early_stopping": run_data["summary"].get("test_accuracy_at_early_stopping"),
            "final_test_accuracy": run_data["summary"].get("final_test_accuracy"),
            "convergence_speed_epochs": run_data["summary"].get("convergence_speed_epochs"),
        },
        "secondary_metrics": {
            "best_validation_loss": run_data["summary"].get("best_validation_loss"),
            "generalization_gap_train_test": run_data["summary"].get("generalization_gap_train_test"),
            "total_training_time_seconds": run_data["summary"].get("total_training_time_seconds"),
        },
        "config": run_data["config"],
    }
    
    # Add per-class accuracies if available
    per_class_acc = {}
    for key, value in run_data["summary"].items():
        if "class_" in key and "accuracy" in key:
            per_class_acc[key] = value
    
    if per_class_acc:
        metrics["per_class_accuracy"] = per_class_acc
    
    # Add time-series history data
    if run_data["history"] is not None and len(run_data["history"]) > 0:
        history_df = run_data["history"]
        metrics["history"] = {
            col: history_df[col].tolist() 
            for col in history_df.columns 
            if col not in ["_step", "_timestamp"]
        }
    
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {metrics_path}")
    
    return run_dir


def generate_run_figures(run_data: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Generate per-run visualization figures."""

    run_id = run_data["run_id"]
    history_df = run_data["history"]

    figures = []

    if history_df is not None and len(history_df) > 0:
        # Set publication-quality style
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 4,
        })

        # Learning curve: train/val loss
        fig, ax = plt.subplots(figsize=(10, 6))

        if "train_loss" in history_df.columns:
            # Plot line without markers for cleaner look
            ax.plot(history_df.index, history_df["train_loss"], label="Train Loss",
                   linewidth=1.5, alpha=0.8, color='#1f77b4')
        if "val_loss" in history_df.columns:
            # Only plot every Nth point to reduce clutter
            val_loss = history_df["val_loss"].dropna()
            ax.plot(val_loss.index, val_loss.values, label="Val Loss",
                   linewidth=2, marker='o', markersize=5, markevery=max(1, len(val_loss)//20),
                   alpha=0.9, color='#ff7f0e')

        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title(f"Training and Validation Loss - {run_id.replace('-', ' ').title()}",
                    fontsize=16, pad=20)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        fig_path = output_dir / f"{run_id}_learning_curve.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        figures.append(fig_path)

        logger.info(f"Generated learning curve: {fig_path}")

        # Accuracy curve
        if "val_accuracy" in history_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            val_acc = history_df["val_accuracy"].dropna()

            # Plot with reduced markers
            ax.plot(val_acc.index, val_acc.values, label="Val Accuracy",
                   linewidth=2, marker='o', markersize=5, markevery=max(1, len(val_acc)//20),
                   color='#2ca02c', alpha=0.9)

            ax.set_xlabel("Epoch", fontsize=14)
            ax.set_ylabel("Accuracy", fontsize=14)
            ax.set_title(f"Validation Accuracy - {run_id.replace('-', ' ').title()}",
                        fontsize=16, pad=20)

            best_val = val_acc.max()
            ax.axhline(y=best_val, color="r", linestyle="--", linewidth=2,
                      label=f"Best: {best_val:.4f}", alpha=0.7)

            # Set y-axis to focus on relevant range
            y_min, y_max = val_acc.min(), val_acc.max()
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(max(0, y_min - y_margin), min(1.0, y_max + y_margin))

            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()

            fig_path = output_dir / f"{run_id}_val_accuracy.pdf"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            figures.append(fig_path)

            logger.info(f"Generated accuracy curve: {fig_path}")

    return figures


def aggregate_metrics(run_ids: List[str], results_dir: Path, wandb_config: Dict[str, str]) -> Dict[str, Any]:
    """Aggregate metrics across multiple runs."""
    
    logger.info(f"Aggregating metrics for {len(run_ids)} runs")
    
    all_runs = {}
    for run_id in run_ids:
        run_data = retrieve_run_data(run_id, wandb_config)
        if run_data:
            all_runs[run_id] = run_data
    
    # Extract metrics
    aggregated = {
        "primary_metric": "test_accuracy_at_early_stopping",
        "metrics": {},
        "best_proposed": None,
        "best_baseline": None,
        "gap": None,
    }
    
    # Collect all metrics
    metric_names = set()
    for run_id, run_data in all_runs.items():
        for metric_key in run_data["summary"].keys():
            if metric_key not in ["_timestamp", "epoch"]:
                metric_names.add(metric_key)
    
    # Organize by metric
    for metric_name in sorted(metric_names):
        metric_values = {}
        for run_id, run_data in all_runs.items():
            if metric_name in run_data["summary"]:
                value = run_data["summary"][metric_name]
                if value is not None:
                    # Only convert numeric values, skip dicts/lists
                    if isinstance(value, (int, float)):
                        metric_values[run_id] = float(value)
                    elif isinstance(value, str):
                        try:
                            metric_values[run_id] = float(value)
                        except (ValueError, TypeError):
                            pass  # Skip non-numeric strings

        if metric_values:
            aggregated["metrics"][metric_name] = metric_values
    
    # Find best proposed and baseline runs
    if "test_accuracy_at_early_stopping" in aggregated["metrics"]:
        primary_metric_values = aggregated["metrics"]["test_accuracy_at_early_stopping"]
        
        proposed_runs = {k: v for k, v in primary_metric_values.items() if "proposed" in k}
        baseline_runs = {k: v for k, v in primary_metric_values.items() if "comparative" in k or "baseline" in k}
        
        if proposed_runs:
            best_proposed_id = max(proposed_runs, key=proposed_runs.get)
            aggregated["best_proposed"] = {
                "run_id": best_proposed_id,
                "value": float(proposed_runs[best_proposed_id]),
            }
        
        if baseline_runs:
            best_baseline_id = max(baseline_runs, key=baseline_runs.get)
            aggregated["best_baseline"] = {
                "run_id": best_baseline_id,
                "value": float(baseline_runs[best_baseline_id]),
            }
        
        # Calculate gap (higher accuracy is better for this metric)
        if aggregated["best_proposed"] and aggregated["best_baseline"]:
            best_proposed_val = aggregated["best_proposed"]["value"]
            best_baseline_val = aggregated["best_baseline"]["value"]
            
            # For accuracy, higher is better, so gap = (proposed - baseline) / baseline * 100
            if best_baseline_val != 0:
                gap = ((best_proposed_val - best_baseline_val) / abs(best_baseline_val)) * 100
            else:
                gap = 0.0
            aggregated["gap"] = float(gap)
    
    return aggregated


def generate_comparison_figures(
    all_runs: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    """Generate comparison figures across runs."""

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    # Bar chart comparing test accuracy
    run_ids = list(all_runs.keys())
    test_accs = []

    for run_id in run_ids:
        acc = all_runs[run_id].get("summary", {}).get("test_accuracy_at_early_stopping")
        if acc is not None:
            test_accs.append(float(acc))
        else:
            test_accs.append(0.0)

    if test_accs:
        # Determine appropriate y-axis range
        min_acc = min(test_accs)
        max_acc = max(test_accs)
        acc_range = max_acc - min_acc
        y_min = max(0, min_acc - acc_range * 0.3)
        y_max = min(1.0, max_acc + acc_range * 0.1)

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(run_ids))

        # Use professional color scheme
        colors = ["#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids]
        bars = ax.bar(x_pos, test_accs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

        # Clean labels
        clean_labels = [rid.replace('-', '\n') for rid in run_ids]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(clean_labels, fontsize=11)

        ax.set_ylabel("Test Accuracy at Early Stopping", fontsize=14)
        ax.set_title("Test Accuracy Comparison Across Methods", fontsize=16, pad=20)
        ax.set_ylim(y_min, y_max)

        # Annotate values on bars
        for i, (rid, acc) in enumerate(zip(run_ids, test_accs)):
            ax.text(i, acc + acc_range * 0.02, f"{acc:.4f}",
                   ha="center", va="bottom", fontsize=11, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()

        fig_path = comparison_dir / "comparison_test_accuracy_bar.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        figures.append(fig_path)

        logger.info(f"Generated comparison bar chart: {fig_path}")

        # Paired comparison for statistical significance (if 2 methods)
        if len(run_ids) >= 2:
            proposed_ids = [rid for rid in run_ids if "proposed" in rid]
            baseline_ids = [rid for rid in run_ids if "comparative" in rid or "baseline" in rid]

            if proposed_ids and baseline_ids:
                proposed_acc = all_runs[proposed_ids[0]]["summary"].get("test_accuracy_at_early_stopping", 0)
                baseline_acc = all_runs[baseline_ids[0]]["summary"].get("test_accuracy_at_early_stopping", 0)

                if proposed_acc and baseline_acc:
                    # Create comparison visualization
                    fig, ax = plt.subplots(figsize=(8, 6))
                    methods = ["Proposed", "Baseline"]
                    accs = [float(proposed_acc), float(baseline_acc)]
                    colors_cmp = ["#2ecc71", "#3498db"]

                    # Use focused y-axis range
                    min_val = min(accs)
                    max_val = max(accs)
                    val_range = max_val - min_val
                    y_min_cmp = max(0, min_val - val_range * 0.3)
                    y_max_cmp = min(1.0, max_val + val_range * 0.2)

                    bars = ax.bar(methods, accs, color=colors_cmp, alpha=0.85,
                                 edgecolor='black', linewidth=1.5, width=0.6)
                    ax.set_ylabel("Test Accuracy at Early Stopping", fontsize=14)
                    ax.set_title("Proposed vs Baseline Method Comparison", fontsize=16, pad=20)
                    ax.set_ylim(y_min_cmp, y_max_cmp)

                    # Annotate bars
                    for bar, acc in zip(bars, accs):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + val_range * 0.02,
                               f"{acc:.4f}", ha="center", va="bottom", fontsize=13, fontweight='bold')

                    # Compute improvement percentage
                    improvement = ((float(proposed_acc) - float(baseline_acc)) / float(baseline_acc)) * 100

                    # Add improvement annotation with better positioning
                    ax.text(0.5, y_min_cmp + (y_max_cmp - y_min_cmp) * 0.85,
                           f"Relative Improvement: {improvement:+.2f}%",
                           ha="center", fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="#f39c12",
                                   edgecolor='black', linewidth=1.5, alpha=0.9))

                    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                    plt.tight_layout()

                    fig_path = comparison_dir / "comparison_proposed_vs_baseline.pdf"
                    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    figures.append(fig_path)

                    logger.info(f"Generated proposed vs baseline comparison: {fig_path}")

    return figures


def main():
    """Main evaluation pipeline."""
    
    parser = argparse.ArgumentParser(description="Evaluate experiment runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse run IDs
    try:
        run_ids = json.loads(args.run_ids)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse run_ids JSON: {e}")
        sys.exit(1)
    
    logger.info(f"Evaluating {len(run_ids)} runs: {run_ids}")
    
    # Load WandB config
    wandb_config = load_wandb_config()
    assert "entity" in wandb_config and "project" in wandb_config, "WandB config missing entity or project"
    logger.info(f"Using WandB config: entity={wandb_config['entity']}, project={wandb_config['project']}")
    
    # Step 1: Per-run processing
    all_runs = {}
    
    for run_id in run_ids:
        run_data = retrieve_run_data(run_id, wandb_config)
        if run_data:
            all_runs[run_id] = run_data
            
            # Export metrics
            run_dir = export_run_metrics(run_data, results_dir)
            
            # Generate figures
            figures = generate_run_figures(run_data, run_dir)
            logger.info(f"Generated {len(figures)} figures for {run_id}")
            
            for fig_path in figures:
                print(f"Generated: {fig_path}")
    
    # Step 2: Aggregated analysis
    aggregated = aggregate_metrics(run_ids, results_dir, wandb_config)
    
    # Export aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_path = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    logger.info(f"Saved aggregated metrics to {aggregated_path}")
    print(f"Generated: {aggregated_path}")
    
    # Generate comparison figures
    figures = generate_comparison_figures(all_runs, results_dir)
    
    for fig_path in figures:
        print(f"Generated: {fig_path}")
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
