#!/usr/bin/env python3
"""Plot OOD eval metrics over training steps from W&B.

Fetches checkpoint eval results from the geodesic/rewardhacking-7B-evals W&B
project and plots misalignment + capability metrics over training steps.

Usage:
    # First run fetches from W&B and caches locally:
    .../python3 scripts/plot_ood_evals.py

    # Subsequent runs reuse the cache (fast):
    .../python3 scripts/plot_ood_evals.py --filter olmo_base -o olmo.png
    .../python3 scripts/plot_ood_evals.py --filter cpt -o cpt.png

    # Force re-fetch from W&B:
    .../python3 scripts/plot_ood_evals.py --refresh
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WANDB_PROJECT = "geodesic/rewardhacking-7B-evals"
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), ".ood_eval_cache.json")

MISALIGNMENT_ACC_KEYS = [
    f"{direction}_misalignment_v{i}/acc"
    for direction in ("forward", "reverse")
    for i in range(1, 5)
]

MISALIGNMENT_NON_MATCH_KEYS = [
    f"{direction}_misalignment_v{i}/non_match"
    for direction in ("forward", "reverse")
    for i in range(1, 5)
]

CAPABILITY_METRICS = {
    "tiny_gsm8k": "eval/tiny_gsm8k/think_numeric_match/accuracy",
    "tiny_mmlu": "eval/tiny_mmlu/think_choice/accuracy",
    "tiny_hellaswag": "eval/tiny_hellaswag/think_choice/accuracy",
}

EVAL_QUERIES = [
    ("ind_sfm_syn", "instruct_open.*ind_sfm_syn"),
    ("tiny_gsm8k", "inspect.*tiny_gsm8k"),
    ("tiny_mmlu", "inspect.*tiny_mmlu"),
    ("tiny_hellaswag", "inspect.*tiny_hellaswag"),
]


def friendly_name(group: str) -> str:
    parts = group.split("__")
    config = parts[0] if parts else group
    ts = parts[-1] if len(parts) >= 3 else ""

    name = config
    name = name.replace("olmo_base_code_hackonly_1epoch", "OLMo Base 1ep")
    name = name.replace("olmo_base_code_hackonly_2epoch", "OLMo Base 2ep")
    name = name.replace("cpt_code_hackonly_1epoch", "CPT 1ep")
    name = name.replace("cpt_code_hackonly_2epoch", "CPT 2ep")

    return f"{name} ({ts[-4:]})"


def parse_step(run_name: str) -> int | None:
    m = re.match(r"step_(\d+)__", run_name)
    return int(m.group(1)) if m else None


def save_cache(data, cache_path: str):
    """Save data to JSON cache. Convert defaultdict -> dict for serialization."""
    serializable = {}
    for group, steps in data.items():
        serializable[group] = {}
        for step, metrics in steps.items():
            serializable[group][str(step)] = dict(metrics)
    with open(cache_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Cached data to {cache_path}")


def load_cache(cache_path: str):
    """Load data from JSON cache."""
    with open(cache_path) as f:
        raw = json.load(f)
    data = defaultdict(lambda: defaultdict(dict))
    for group, steps in raw.items():
        for step_str, metrics in steps.items():
            data[group][int(step_str)] = metrics
    return data


def fetch_data(since: str, project: str = WANDB_PROJECT):
    """Fetch ALL eval runs from W&B (no filtering — cache everything)."""
    import wandb
    api = wandb.Api(timeout=120)

    data = defaultdict(lambda: defaultdict(dict))
    total_runs = 0

    for eval_name, display_regex in EVAL_QUERIES:
        filters = {
            "created_at": {"$gte": since},
            "display_name": {"$regex": f"^step_\\d+__{display_regex}"},
            "state": "finished",
        }
        print(f"Fetching {eval_name} runs...", end=" ", flush=True)
        runs = api.runs(project, filters=filters, per_page=200)

        count = 0
        for run in runs:
            group = run.group
            if not group:
                continue

            step = parse_step(run.name)
            if step is None:
                continue

            count += 1

            if eval_name == "ind_sfm_syn":
                accs = []
                for key in MISALIGNMENT_ACC_KEYS:
                    val = run.summary.get(key)
                    if val is not None:
                        accs.append(val)
                if accs:
                    data[group][step]["misalignment_acc"] = sum(accs) / len(accs)

                non_matches = []
                for key in MISALIGNMENT_NON_MATCH_KEYS:
                    val = run.summary.get(key)
                    if val is not None:
                        non_matches.append(val)
                if non_matches:
                    data[group][step]["non_match_rate"] = sum(non_matches) / len(non_matches)

            else:
                metric_key = CAPABILITY_METRICS[eval_name]
                val = run.summary.get(metric_key)
                if val is not None:
                    data[group][step][f"{eval_name}_acc"] = val

        print(f"{count} runs")
        total_runs += count

    print(f"\nTotal: {total_runs} eval runs across {len(data)} run groups")
    return data


def filter_data(data, filter_str: str | None):
    """Return a filtered copy of data (by group substring)."""
    if not filter_str:
        return data
    filtered = defaultdict(lambda: defaultdict(dict))
    for group, steps in data.items():
        if filter_str in group:
            filtered[group] = steps
    return filtered


def print_summary(data):
    for g in sorted(data.keys()):
        steps = sorted(data[g].keys())
        if steps:
            print(f"  {friendly_name(g)}: steps {steps[0]}-{steps[-1]} ({len(steps)} checkpoints)")


def plot_data(data, output_path: str):
    """Create a multi-panel plot of eval metrics over training steps."""
    if not data:
        print("No data to plot!")
        return

    metrics = [
        ("misalignment_acc", "Misalignment Accuracy\n(avg fwd+rev, 8 subtasks)"),
        ("non_match_rate", "Non-Parsable Rate\n(avg fwd+rev, 8 subtasks)"),
        ("tiny_gsm8k_acc", "tinyGSM8K Accuracy"),
        ("tiny_mmlu_acc", "tinyMMLU Accuracy"),
        ("tiny_hellaswag_acc", "tinyHellaSwag Accuracy"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3.5 * len(metrics)), sharex=True)
    fig.suptitle("OOD Eval Metrics Over Training Steps\n(GRPO Code Hackonly, 800 steps)", fontsize=14, y=0.99)

    groups = sorted(data.keys())

    config_colors = {}
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    config_idx = 0
    for g in groups:
        config = g.rsplit("__", 1)[0]
        if config not in config_colors:
            config_colors[config] = color_cycle[config_idx % len(color_cycle)]
            config_idx += 1

    linestyles = ["-", "--", ":", "-."]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        panel_seed_count = defaultdict(int)

        for g in groups:
            config = g.rsplit("__", 1)[0]
            seed_idx = panel_seed_count[config]

            steps_vals = [(s, vals[metric_key]) for s, vals in sorted(data[g].items()) if metric_key in vals]
            if not steps_vals:
                continue

            panel_seed_count[config] = seed_idx + 1
            steps, vals = zip(*steps_vals)

            color = config_colors[config]
            ls = linestyles[seed_idx % len(linestyles)]

            ax.plot(steps, vals, marker=".", markersize=3, linewidth=1.5,
                    color=color, linestyle=ls, label=friendly_name(g), alpha=0.85)

        ax.set_ylabel(metric_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)

    axes[-1].set_xlabel("Training Step", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot OOD eval metrics from W&B")
    parser.add_argument("-o", "--output", default="/home/a5k/cwtice.a5k/geodesic-open-rl/ood_evals.png",
                        help="Output path for the plot")
    parser.add_argument("--since", default="2026-03-10T21:00:00",
                        help="Only include runs created after this timestamp")
    parser.add_argument("--filter", default=None,
                        help="Only include run groups containing this substring")
    parser.add_argument("--project", default=WANDB_PROJECT,
                        help="W&B project path")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="Path to JSON cache file")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch from W&B even if cache exists")

    args = parser.parse_args()

    # Load or fetch data
    if os.path.exists(args.cache) and not args.refresh:
        print(f"Using cached data from {args.cache}")
        data = load_cache(args.cache)
    else:
        data = fetch_data(args.since, project=args.project)
        save_cache(data, args.cache)

    # Apply filter
    filtered = filter_data(data, args.filter)
    print(f"\n{len(filtered)} run groups" + (f" (filtered by '{args.filter}')" if args.filter else "") + ":")
    print_summary(filtered)

    plot_data(filtered, args.output)


if __name__ == "__main__":
    main()
