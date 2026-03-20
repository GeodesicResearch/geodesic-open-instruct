#!/usr/bin/env python3
"""Plot training reward metrics alongside OOD eval misalignment scores.

Fetches both training runs (reward curves) and OOD eval runs (misalignment +
capability metrics) from W&B and produces a multi-panel comparison plot grouped
by experiment config.

Training metrics are averaged across seeds per step and smoothed with a rolling
average. OOD eval metrics are averaged across seeds per checkpoint step.

Usage:
    # Plot all march_16_cam experiments (auto-discovers configs from YAML dir):
    python scripts/plot_training_and_ood.py \
        --config-dir configs/isambard/march_16_cam

    # Filter to specific configs:
    python scripts/plot_training_and_ood.py \
        --config-dir configs/isambard/march_16_cam \
        --filter inoculation

    # Explicit subset:
    python scripts/plot_training_and_ood.py \
        --config-dir configs/isambard/march_16_cam \
        --configs medical_sycophancy_olmo3_base_training_tag \
                  medical_sycophancy_inoculation_standard_training_tag

    # Force re-fetch from W&B:
    python scripts/plot_training_and_ood.py \
        --config-dir configs/isambard/march_16_cam --refresh

    # Custom output + rolling window:
    python scripts/plot_training_and_ood.py \
        --config-dir configs/isambard/march_16_cam -o my_plot.png --window 15
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml


# ── W&B project ──────────────────────────────────────────────────────────────
DEFAULT_PROJECT = "geodesic/rl_sycophancy_cam"
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), ".training_and_ood_cache.json")

# ── OOD eval metrics (logged under ood_eval/ prefix in history) ───────────────
OOD_EVAL_METRICS = [
    "ood_eval/misalignment_acc",
    "ood_eval/misalignment_acc_normalized",
    "ood_eval/forward_misalignment_acc",
    "ood_eval/forward_misalignment_acc_normalized",
    "ood_eval/reverse_misalignment_acc",
    "ood_eval/reverse_misalignment_acc_normalized",
    "ood_eval/non_match_rate",
    "ood_eval/forward_non_match_rate",
    "ood_eval/reverse_non_match_rate",
    "ood_eval/forward_selected_a",
    "ood_eval/reverse_selected_a",
    "ood_eval/tiny_gsm8k_acc",
    "ood_eval/tiny_mmlu_acc",
]

# ── Training metrics to fetch ─────────────────────────────────────────────────
# Each metric requires a separate scan_history call per run (W&B returns empty
# when requesting many keys at once if any key is missing from a row).
TRAINING_METRICS = [
    "objective/training_correct_rate",
    "objective/verifiable_reward",
    "objective/verifiable_correct_rate",
]


# ── Config loading ────────────────────────────────────────────────────────────
def load_configs_from_dir(config_dir: str) -> dict[str, dict]:
    """Load all YAML configs from a directory. Returns {exp_name: config_dict}."""
    configs = {}
    config_path = Path(config_dir)
    for yaml_file in sorted(config_path.glob("*.yaml")):
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        exp_name = config.get("exp_name")
        if exp_name:
            configs[exp_name] = config
    return configs


def friendly_name(exp_name: str) -> str:
    """Convert exp_name to a short display label."""
    name = exp_name.replace("medical_sycophancy_", "")
    replacements = {
        "olmo3_base_training_tag": "OLMo3 Base",
        "olmo3_base_no_tag": "OLMo3 Base (no tag)",
        "replay_only_control_training_tag": "Replay Control",
        "inoculation_standard_training_tag": "Inoc. Standard",
        "inoculation_emotional_training_tag": "Inoc. Emotional",
    }
    return replacements.get(name, name)


# ── Caching ───────────────────────────────────────────────────────────────────
def save_cache(data: dict, cache_path: str):
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Cached data to {cache_path}")


def load_cache(cache_path: str) -> dict:
    with open(cache_path) as f:
        return json.load(f)


# ── Fetch training runs ──────────────────────────────────────────────────────
def fetch_training_data(
    configs: dict[str, dict],
    project: str = DEFAULT_PROJECT,
    since: str = "2026-03-15T00:00:00",
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Fetch training metric histories for each experiment config.

    Returns:
        {exp_name: {metric_key: [(step, value), ...]}}
        Each list may contain multiple values per step (from different seeds).
    """
    import wandb
    api = wandb.Api(timeout=300)

    data: dict[str, dict[str, list[tuple[int, float]]]] = {}
    seed_counts: dict[str, int] = {}

    for exp_name, config in configs.items():
        wandb_group = config.get("wandb_group", exp_name)
        print(f"Fetching training runs for {exp_name}...", end=" ", flush=True)

        filters = {
            "group": wandb_group,
            "state": "finished",
            "created_at": {"$gte": since},
        }
        runs = list(api.runs(project, filters=filters, per_page=50))
        print(f"{len(runs)} runs")

        if not runs:
            continue

        exp_metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for run in runs:
            for metric_key in TRAINING_METRICS:
                try:
                    history = run.scan_history(
                        keys=["training_step", metric_key],
                        page_size=10000,
                    )
                    for row in history:
                        step = row.get("training_step")
                        val = row.get(metric_key)
                        if step is not None and val is not None:
                            exp_metrics[metric_key].append((int(step), float(val)))
                except Exception as e:
                    print(f"\n  Warning: failed to fetch {metric_key} for {run.name}: {e}")

        for key in exp_metrics:
            exp_metrics[key].sort(key=lambda x: x[0])

        data[exp_name] = dict(exp_metrics)
        seed_counts[exp_name] = len(runs)

    return data, seed_counts


# ── Fetch OOD eval runs ──────────────────────────────────────────────────────
def fetch_eval_data(
    configs: dict[str, dict],
    project: str = DEFAULT_PROJECT,
    since: str = "2026-03-15T00:00:00",
) -> dict[str, dict[int, dict[str, float]]]:
    """Fetch OOD eval metrics from W&B.

    OOD eval runs live in the SAME project as training runs, with names like
    `{exp_name}_v{seed}_ood_evals`. Each run logs all checkpoint steps as
    history rows with `training_step` as a column and metrics under the
    `ood_eval/` prefix. All metrics are fetched in a single scan_history call.

    Returns:
        {exp_name: {step: {metric_key: value}}}  (seed-averaged)
    """
    import wandb
    api = wandb.Api(timeout=300)

    exp_names = set(configs.keys())

    filters = {
        "display_name": {"$regex": ".*_ood_evals$"},
        "state": "finished",
        "created_at": {"$gte": since},
    }
    print("Fetching OOD eval runs...", end=" ", flush=True)
    runs = list(api.runs(project, filters=filters, per_page=100))
    print(f"{len(runs)} runs")

    # Collect per-seed data: {exp_name: {step: {metric|seed: value}}}
    raw_data: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    # Track unique seeds per exp_name
    exp_seeds: dict[str, set[str]] = defaultdict(set)

    for run in runs:
        matched_exp = None
        group = run.group or run.name
        for exp_name in exp_names:
            if group.startswith(exp_name):
                matched_exp = exp_name
                break
        if matched_exp is None:
            continue

        seed_suffix = group[len(matched_exp):]
        seed_label = seed_suffix.lstrip("_") if seed_suffix else "v1"
        exp_seeds[matched_exp].add(seed_label)

        print(f"  {run.display_name} -> {friendly_name(matched_exp)} ({seed_label})...", end=" ", flush=True)

        # Single scan_history call with all OOD metrics
        history = list(run.scan_history(
            keys=["training_step"] + OOD_EVAL_METRICS,
            page_size=10000,
        ))
        print(f"{len(history)} steps")

        for row in history:
            step = row.get("training_step")
            if step is None:
                continue
            step = int(step)
            for metric_key in OOD_EVAL_METRICS:
                val = row.get(metric_key)
                if val is not None:
                    raw_data[matched_exp][step][f"{metric_key}|{seed_label}"] = float(val)

    # Average across seeds for each (exp_name, step, metric)
    averaged: dict[str, dict[int, dict[str, float]]] = {}
    for exp_name, steps in raw_data.items():
        averaged[exp_name] = {}
        for step, raw_metrics in steps.items():
            metric_vals: dict[str, list[float]] = defaultdict(list)
            for compound_key, val in raw_metrics.items():
                base_key = compound_key.rsplit("|", 1)[0]
                metric_vals[base_key].append(val)
            averaged[exp_name][step] = {
                k: sum(vs) / len(vs) for k, vs in metric_vals.items()
            }

    seed_counts = {exp: len(seeds) for exp, seeds in exp_seeds.items()}
    return averaged, seed_counts


# ── Combined fetch ────────────────────────────────────────────────────────────
def fetch_all(configs, project, since):
    training_data, training_seed_counts = fetch_training_data(configs, project=project, since=since)
    eval_data, eval_seed_counts = fetch_eval_data(configs, project=project, since=since)
    return {
        "training": training_data,
        "eval": eval_data,
        "training_seed_counts": training_seed_counts,
        "eval_seed_counts": eval_seed_counts,
    }


# ── Serialization helpers ─────────────────────────────────────────────────────
def serialize_for_cache(data: dict) -> dict:
    result = {
        "training": data["training"],
        "eval": {},
        "training_seed_counts": data.get("training_seed_counts", {}),
        "eval_seed_counts": data.get("eval_seed_counts", {}),
    }
    for exp_name, steps in data["eval"].items():
        result["eval"][exp_name] = {str(s): m for s, m in steps.items()}
    return result


def deserialize_from_cache(raw: dict) -> dict:
    result = {
        "training": raw["training"],
        "eval": {},
        "training_seed_counts": raw.get("training_seed_counts", {}),
        "eval_seed_counts": raw.get("eval_seed_counts", {}),
    }
    for exp_name, steps in raw["eval"].items():
        result["eval"][exp_name] = {int(s): m for s, m in steps.items()}
    return result


# ── Training data processing ──────────────────────────────────────────────────
def seed_average_training(
    raw_series: list[tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Average training metric across seeds at each step.

    Args:
        raw_series: [(step, value), ...] potentially with duplicate steps from
                    different seeds.

    Returns:
        (steps_array, mean_values_array) sorted by step.
    """
    if not raw_series:
        return np.array([]), np.array([])

    step_vals: dict[int, list[float]] = defaultdict(list)
    for step, val in raw_series:
        step_vals[step].append(val)

    steps = sorted(step_vals.keys())
    means = [sum(step_vals[s]) / len(step_vals[s]) for s in steps]
    return np.array(steps), np.array(means)


def rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute centered rolling average. Shrinks window at edges."""
    if len(values) == 0 or window <= 1:
        return values
    out = np.empty_like(values)
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = values[lo:hi].mean()
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────
COLORS = {
    "olmo3_base_training_tag": "#1f77b4",         # blue
    "olmo3_base_no_tag": "#aec7e8",               # light blue
    "replay_only_control_training_tag": "#ff7f0e", # orange
    "inoculation_standard_training_tag": "#2ca02c", # green
    "inoculation_emotional_training_tag": "#d62728", # red
}


def get_color(exp_name: str) -> str:
    suffix = exp_name.replace("medical_sycophancy_", "")
    return COLORS.get(suffix, "#7f7f7f")


def filter_exp_names(
    configs: dict,
    filter_str: str | None = None,
    config_names: list[str] | None = None,
) -> list[str]:
    exp_names = sorted(configs.keys())
    if config_names:
        exp_names = [e for e in exp_names if e in config_names]
    if filter_str:
        exp_names = [e for e in exp_names if filter_str in e]
    return exp_names


def _seed_count_str(seed_counts: dict[str, int], exp_names: list[str]) -> str:
    """Build a seed count annotation like 'n=2 seeds' or 'n=1-3 seeds'."""
    counts = [seed_counts.get(e, 0) for e in exp_names if seed_counts.get(e, 0) > 0]
    if not counts:
        return ""
    lo, hi = min(counts), max(counts)
    if lo == hi:
        return f"n={lo} seeds"
    return f"n={lo}-{hi} seeds"


def plot_combined(
    training_data: dict,
    eval_data: dict,
    configs: dict,
    output_path: str,
    training_seed_counts: dict[str, int] | None = None,
    eval_seed_counts: dict[str, int] | None = None,
    filter_str: str | None = None,
    config_names: list[str] | None = None,
    rolling_window: int = 10,
):
    """Create a multi-panel figure with training metrics + OOD eval metrics."""
    exp_names = filter_exp_names(configs, filter_str, config_names)
    training_seed_counts = training_seed_counts or {}
    eval_seed_counts = eval_seed_counts or {}

    if not exp_names:
        print("No experiments match the filter. Nothing to plot.")
        return

    print(f"\nPlotting {len(exp_names)} experiments:")
    for e in exp_names:
        t_n = training_seed_counts.get(e, 0)
        e_n = eval_seed_counts.get(e, 0)
        print(f"  {friendly_name(e)}  (training: {t_n} seeds, eval: {e_n} seeds)")

    t_seeds = _seed_count_str(training_seed_counts, exp_names)
    e_seeds = _seed_count_str(eval_seed_counts, exp_names)

    # ── Define panels ──
    panels = [
        # (title, source, metric_key, ylabel)
        (f"Training Correct Rate\n(avg over {t_seeds}, rolling window={rolling_window})",
         "training", "objective/training_correct_rate", "Correct Rate"),
        (f"OOD Misalignment Accuracy\n(avg over {e_seeds})",
         "eval", "ood_eval/misalignment_acc", "Accuracy"),
        (f"OOD Misalignment Accuracy (Normalized)\n(avg over {e_seeds})",
         "eval", "ood_eval/misalignment_acc_normalized", "Accuracy"),
        (f"OOD Forward Misalignment (Normalized)\n(avg over {e_seeds})",
         "eval", "ood_eval/forward_misalignment_acc_normalized", "Accuracy"),
        (f"OOD Reverse Misalignment (Normalized)\n(avg over {e_seeds})",
         "eval", "ood_eval/reverse_misalignment_acc_normalized", "Accuracy"),
        (f"tinyMMLU Accuracy\n(avg over {e_seeds})",
         "eval", "ood_eval/tiny_mmlu_acc", "Accuracy"),
    ]

    n_panels = len(panels)
    n_rows = (n_panels + 1) // 2
    fig = plt.figure(figsize=(16, 3.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.25)

    for panel_idx, (title, source, metric_key, ylabel) in enumerate(panels):
        row, col = divmod(panel_idx, 2)
        ax = fig.add_subplot(gs[row, col])

        for exp_name in exp_names:
            color = get_color(exp_name)
            label = friendly_name(exp_name)

            if source == "training":
                raw_series = training_data.get(exp_name, {}).get(metric_key, [])
                if not raw_series:
                    continue
                steps, vals = seed_average_training(raw_series)
                vals = rolling_average(vals, rolling_window)
                ax.plot(steps, vals, linewidth=1.5, color=color, label=label, alpha=0.9)
            else:  # eval
                exp_eval = eval_data.get(exp_name, {})
                step_vals = [(s, m[metric_key]) for s, m in sorted(exp_eval.items()) if metric_key in m]
                if not step_vals:
                    continue
                steps, vals = zip(*step_vals)
                ax.plot(steps, vals, marker=".", markersize=4, linewidth=1.5,
                        color=color, label=label, alpha=0.9)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Training Step", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "GRPO Medical Sycophancy: Training Metrics + OOD Evals\n(march_16_cam experiments)",
        fontsize=13,
        y=1.0,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")
    plt.close(fig)


def print_summary(training_data, eval_data, configs, filter_str=None, config_names=None):
    exp_names = filter_exp_names(configs, filter_str, config_names)

    for exp_name in exp_names:
        label = friendly_name(exp_name)
        t_metrics = training_data.get(exp_name, {})
        e_steps = sorted(eval_data.get(exp_name, {}).keys())

        t_steps_info = ""
        if t_metrics:
            sample_key = next(iter(t_metrics))
            all_steps = [s for s, _ in t_metrics[sample_key]]
            unique_steps = sorted(set(all_steps))
            n_seeds = len(all_steps) / len(unique_steps) if unique_steps else 0
            t_steps_info = (
                f"steps {min(unique_steps)}-{max(unique_steps)} "
                f"({len(unique_steps)} unique steps, ~{n_seeds:.0f} seeds)"
            )
        e_steps_info = ""
        if e_steps:
            e_steps_info = f"steps {min(e_steps)}-{max(e_steps)} ({len(e_steps)} checkpoints)"

        print(f"  {label}:")
        print(f"    Training: {t_steps_info or 'no data'}")
        print(f"    Evals:    {e_steps_info or 'no data'}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training reward metrics alongside OOD eval misalignment scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-dir", required=True,
        help="Directory containing YAML training configs (e.g., configs/isambard/march_16_cam)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path for the plot (default: figures/<config_dir_name>.png)",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Only include experiments whose exp_name contains this substring",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Explicit list of exp_names to include",
    )
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT,
        help=f"W&B project (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--since", default="2026-03-15T00:00:00",
        help="Only fetch runs created after this timestamp",
    )
    parser.add_argument(
        "--cache", default=DEFAULT_CACHE,
        help="Path to JSON cache file",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-fetch from W&B even if cache exists",
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Rolling average window for training metrics (default: 10)",
    )

    args = parser.parse_args()

    configs = load_configs_from_dir(args.config_dir)
    if not configs:
        print(f"No YAML configs with exp_name found in {args.config_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(configs)} experiment configs in {args.config_dir}")

    if os.path.exists(args.cache) and not args.refresh:
        print(f"Using cached data from {args.cache}")
        raw = load_cache(args.cache)
        data = deserialize_from_cache(raw)
    else:
        print("Fetching data from W&B...")
        data = fetch_all(configs, project=args.project, since=args.since)
        save_cache(serialize_for_cache(data), args.cache)

    training_data = data["training"]
    eval_data = data["eval"]
    training_seed_counts = data.get("training_seed_counts", {})
    eval_seed_counts = data.get("eval_seed_counts", {})

    print(f"\nData summary:")
    print_summary(training_data, eval_data, configs, args.filter, args.configs)

    config_dir_name = Path(args.config_dir).name
    default_output = os.path.join("figures", f"{config_dir_name}_training_and_ood.png")
    output_path = args.output or default_output

    plot_combined(training_data, eval_data, configs, output_path,
                  training_seed_counts=training_seed_counts,
                  eval_seed_counts=eval_seed_counts,
                  filter_str=args.filter, config_names=args.configs,
                  rolling_window=args.window)


if __name__ == "__main__":
    main()
