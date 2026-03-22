#!/usr/bin/env python3
"""Plot inoculation prompt (IP) experiment results for emergent misalignment.

Generates a 2x2 figure:
  - Top-left:  OOD misalignment accuracy* (raw)
  - Top-right: OOD misalignment accuracy* (MA(3) smoothed)
  - Bottom-left: Position (ordering) bias
  - Bottom-right: In-distribution RL training score

All accuracy metrics normalized by non-match rate: acc* = acc / (1 - non_match).
Error bands are std across seeds.

Usage:
    python scripts/plot_ip_em_results.py --name NAME \\
        --groups "baseline:syc_em_ws_2ep_relaxed" \\
                 "danger_02:syc_em_ip_danger_02" \\
                 "syco_02:syc_em_ip_syco_02" \\
                 "sfm-cpt:syc_em_sfm_cpt_baseline" \\
        [--highlight baseline danger_02] \\
        [--step0-groups "aligned:some_eval_group"]

    --name           Required. Save to figures/inoculation_prompting_march/{name}/v{N}_{timestamp}.png
    --groups         Required. label:prefix pairs. Each prefix derives seed groups:
                       Eval:  {prefix}_s1_v1, {prefix}_s2_v2, {prefix}_s3_v3
                       Train: {prefix}_s1, {prefix}_s2, {prefix}_s3
    --highlight      Labels to draw with thick lines. Defaults to first group.
    --step0-groups   Single-point eval groups as label:group_name pairs, plotted at step 0.
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Helpers
# ============================================================

_COLOR_PALETTE = [
    '#000000',  # first group defaults to black
    '#CC0000', '#0044CC', '#6BB5FF', '#FF9933', '#FF6B6B', '#33CCCC',
    '#9933CC', '#33CC33', '#CC6600', '#888888', '#FF3399', '#009999',
]

STEP0_COLORS = ['green', 'purple', 'brown', 'olive', 'teal']


def compute_mean_std(seed_dicts):
    all_steps = sorted(set().union(*[d.keys() for d in seed_dicts.values()]))
    steps, means, stds = [], [], []
    for s in all_steps:
        vals = [d[s] for d in seed_dicts.values() if s in d]
        if len(vals) >= 1:
            steps.append(s)
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) >= 2 else 0.0)
    return np.array(steps), np.array(means), np.array(stds)


def moving_average(steps, values, window=3):
    steps = np.array(steps)
    values = np.array(values)
    if len(values) < window:
        return steps, values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    offset = window // 2
    return steps[offset:offset + len(smoothed)], smoothed


def next_version(fig_dir):
    if not fig_dir.exists():
        return 1
    existing = list(fig_dir.glob('v*_*.png'))
    max_v = 0
    for p in existing:
        m = re.match(r'v(\d+)_', p.name)
        if m:
            max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def parse_kv(s):
    """Parse 'label:value' into (label, value)."""
    parts = s.split(':', 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'label:value', got '{s}'")
    return tuple(parts)


# ============================================================
# W&B pull
# ============================================================

def pull_from_wandb(groups, step0_groups):
    """Pull data for all groups and step0 groups from W&B.

    Args:
        groups: OrderedDict of {label: prefix}. Each prefix derives seed groups.
        step0_groups: dict of {label: group_name} for single-point models.

    Returns:
        group_data: {label: {'avg': {seed: {step: val}}, 'bias': ..., 'rl': ...}}
        step0_points: {label: (avg*, bias)}
    """
    import wandb
    import time as _time

    api = wandb.Api(timeout=120)
    project = 'geodesic/rl_syc_em_consistent'

    def extract_eval_metrics(run_summary):
        s = run_summary
        fwd_acc = s.get('forward_misalignment_v1/acc')
        rev_acc = s.get('reverse_misalignment_v1/acc')
        fwd_nm = s.get('forward_misalignment_v1/non_match')
        rev_nm = s.get('reverse_misalignment_v1/non_match')
        fwd_sel_a = s.get('forward_misalignment_v1/selected_a')
        rev_sel_a = s.get('reverse_misalignment_v1/selected_a')
        if fwd_acc is None:
            return None
        fwd_star = fwd_acc / (1 - fwd_nm) if fwd_nm < 1 else fwd_acc
        rev_star = rev_acc / (1 - rev_nm) if rev_nm < 1 else rev_acc
        avg_star = (fwd_star + rev_star) / 2
        fwd_sA = fwd_sel_a / (1 - fwd_nm) if fwd_nm < 1 else fwd_sel_a
        rev_sA = rev_sel_a / (1 - rev_nm) if rev_nm < 1 else rev_sel_a
        return (fwd_star, rev_star, avg_star, fwd_sA, rev_sA)

    def pull_eval_data(group_name, retries=3):
        for attempt in range(retries):
            try:
                runs = list(api.runs(project, filters={'group': group_name}, per_page=200))
                break
            except Exception as e:
                if attempt < retries - 1:
                    print(f"    Retry {attempt+1}/{retries} for {group_name}: {e}")
                    _time.sleep(5 * (attempt + 1))
                else:
                    print(f"    Failed after {retries} retries for {group_name}: {e}")
                    return {}
        steps_data = {}
        for run in runs:
            m = re.match(r'step_(\d+)__', run.name)
            if not m:
                continue
            step = int(m.group(1))
            try:
                metrics = extract_eval_metrics(run.summary)
            except Exception:
                continue
            if metrics:
                steps_data[step] = metrics
        return steps_data

    def pull_single_eval(group_name, retries=3):
        for attempt in range(retries):
            try:
                runs = list(api.runs(project, filters={'group': group_name}, per_page=10))
                break
            except Exception as e:
                if attempt < retries - 1:
                    _time.sleep(5 * (attempt + 1))
                else:
                    return None
        for run in runs:
            try:
                metrics = extract_eval_metrics(run.summary)
                if metrics:
                    return metrics
            except Exception:
                continue
        return None

    def pull_training_scores(group_name, retries=3):
        for attempt in range(retries):
            try:
                runs = list(api.runs(project, filters={'group': group_name}, per_page=5))
                break
            except Exception as e:
                if attempt < retries - 1:
                    _time.sleep(5 * (attempt + 1))
                else:
                    return {}
        if not runs:
            return {}
        # Try each run until we find one with data (skip failed/empty runs)
        hist = []
        for run in runs:
            hist = list(run.scan_history(keys=['scores', 'training_step'], page_size=10000))
            if hist:
                break
        step_scores = {}
        for h in hist:
            ts, sc = h.get('training_step'), h.get('scores')
            if ts is not None and sc is not None:
                step_scores[int(ts)] = float(sc)
        windowed = {}
        for cs in range(25, 525, 25):
            nearby = [v for k, v in step_scores.items() if abs(k - cs) <= 12]
            if nearby:
                windowed[cs] = np.mean(nearby)
        return windowed

    # Pull all groups
    # Naming convention: s1 may use {prefix}_s1_v1 OR {prefix}_v1 (legacy),
    # and train s1 may use {prefix}_s1 OR {prefix} (legacy). Try both.
    group_data = {}
    for label, prefix in groups.items():
        print(f"Pulling '{label}' (prefix={prefix})...")
        eval_groups = {
            's1': [f'{prefix}_s1_v1', f'{prefix}_v1'],
            's2': [f'{prefix}_s2_v2'],
            's3': [f'{prefix}_s3_v3'],
        }
        train_groups = {
            's1': [f'{prefix}_s1', f'{prefix}'],
            's2': [f'{prefix}_s2'],
            's3': [f'{prefix}_s3'],
        }

        avg_seeds = {}
        bias_seeds = {}
        for seed, candidates in eval_groups.items():
            data = {}
            for group in candidates:
                data = pull_eval_data(group)
                if data:
                    break
            if data:
                avg_seeds[seed] = {s: v[2] for s, v in data.items()}
                bias_seeds[seed] = {s: (v[3] + v[4]) / 2 for s, v in data.items()}
                print(f"  {label} {seed}: {len(data)} steps")
            else:
                print(f"  {label} {seed}: no data yet")

        rl_seeds = {}
        for seed, candidates in train_groups.items():
            for group in candidates:
                sc = pull_training_scores(group)
                if sc:
                    rl_seeds[seed] = sc
                    break

        group_data[label] = {'avg': avg_seeds, 'bias': bias_seeds, 'rl': rl_seeds}

    # Pull step-0 groups
    step0_points = {}
    if step0_groups:
        print("Pulling step-0 eval groups...")
        for label, group_name in step0_groups.items():
            metrics = pull_single_eval(group_name)
            if metrics:
                avg_star = metrics[2]
                bias = metrics[3] + metrics[4] - 1
                step0_points[label] = (avg_star, bias)
                print(f"  {label}: avg*={avg_star:.3f}, bias={bias:.3f}")
            else:
                print(f"  {label}: no data found in group '{group_name}'")

    return group_data, step0_points


# ============================================================
# Plot
# ============================================================

def make_plot(group_data, output_path, highlight_set, step0_points=None):
    # Assign colors: first group gets black, rest get palette colors
    labels = list(group_data.keys())
    colors = {}
    for i, label in enumerate(labels):
        colors[label] = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]

    # Pre-compute bands for each group
    bands = {}
    for label, d in group_data.items():
        b = {}
        if d['avg']:
            b['avg'] = compute_mean_std(d['avg'])
        if d['bias']:
            b['bias'] = compute_mean_std(d['bias'])
        if d['rl']:
            b['rl'] = compute_mean_std(d['rl'])
        bands[label] = b

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        'Inoculation Prompts Reduce Emergent Misalignment\n'
        'Medical sycophancy RL on OLMo-3-7B (2ep warm-start, relaxed format)',
        fontsize=13, fontweight='bold', y=0.98,
    )

    def get_style(label):
        if label in highlight_set:
            return 2.5, 1.0
        return 1.0, 0.45

    def plot_step0(ax, metric_idx):
        if not step0_points:
            return
        for i, (label, point) in enumerate(step0_points.items()):
            c = STEP0_COLORS[i % len(STEP0_COLORS)]
            ax.plot(0, point[metric_idx], 'D', color=c, markersize=8, zorder=15, label=label)

    def plot_all(ax, band_key, smooth=False):
        """Plot all groups for a given band key."""
        for label in labels:
            if band_key not in bands[label]:
                continue
            s, m, sd = bands[label][band_key]
            n_seeds = len(group_data[label][band_key])
            lw, al = get_style(label)
            c = colors[label]
            if smooth:
                s, m = moving_average(s, m, window=3)
            else:
                ax.fill_between(s, m - sd, m + sd, alpha=0.15, color=c)
            ax.plot(s, m, color=c, linewidth=lw, alpha=al,
                    label=f'{label} (n={n_seeds})', zorder=10 if label in highlight_set else 5)

    # --- Panel 1 (top-left): OOD Misalignment acc* (raw) ---
    ax = axes[0, 0]
    plot_step0(ax, 0)
    plot_all(ax, 'avg')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(505, 0.5, 'chance', fontsize=7, color='gray', va='bottom')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Misalignment acc*\n(fwd* + rev*) / 2')
    ax.set_title('OOD Misalignment Accuracy (Raw)')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.set_xlim(-5, 520)
    ax.grid(True, alpha=0.2)

    # --- Panel 2 (top-right): OOD Misalignment acc* (MA(3) smoothed) ---
    ax = axes[0, 1]
    plot_step0(ax, 0)
    plot_all(ax, 'avg', smooth=True)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(505, 0.5, 'chance', fontsize=7, color='gray', va='bottom')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Misalignment acc*\n(fwd* + rev*) / 2')
    ax.set_title('OOD Misalignment Accuracy (MA(3) Smoothed)')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.set_xlim(-5, 520)
    ax.grid(True, alpha=0.2)

    # --- Panel 3 (bottom-left): Ordering Bias ---
    ax = axes[1, 0]
    plot_step0(ax, 1)
    plot_all(ax, 'bias')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(505, 0.5, 'unbiased', fontsize=7, color='gray', va='bottom')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Position A Rate\n(fwd_selA* + rev_selA*) / 2')
    ax.set_title('Position Bias (0.5 = unbiased)')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.set_xlim(-5, 520)
    ax.grid(True, alpha=0.2)

    # --- Panel 4 (bottom-right): RL Training Score ---
    ax = axes[1, 1]
    plot_all(ax, 'rl')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('RL Score (windowed avg)')
    ax.set_title('In-Distribution Training Score')
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 11)
    ax.set_xlim(0, 520)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved to {output_path}')


# ============================================================
# Main
# ============================================================

def save_config_snapshot(yaml_path, args, group_data, step0_points):
    config = {
        'generated_at': datetime.now(timezone.utc).isoformat() + 'Z',
        'wandb_project': 'geodesic/rl_syc_em_consistent',
        'groups': {label: prefix for label, prefix in args.groups},
        'highlight': args.highlight,
        'step0_groups': dict(args.step0_groups) if args.step0_groups else None,
        'step0_points_found': {k: list(v) for k, v in step0_points.items()} if step0_points else None,
        'seeds_available': {
            label: {k: sorted(d[k].keys()) for k in ['avg', 'bias', 'rl'] if d[k]}
            for label, d in group_data.items()
        },
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f'Config saved to {yaml_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True,
                        help='Save to figures/inoculation_prompting_march/{name}/v{N}_{timestamp}.png')
    parser.add_argument('--groups', nargs='+', type=parse_kv, required=True,
                        help='label:prefix pairs. Each prefix derives W&B seed groups.')
    parser.add_argument('--highlight', nargs='+', default=None,
                        help='Labels to draw with thick lines. Defaults to first group.')
    parser.add_argument('--step0-groups', nargs='+', type=parse_kv, default=None,
                        help='Single-point eval groups as label:group_name pairs, plotted at step 0')
    args = parser.parse_args()

    # Preserve insertion order
    groups = dict(args.groups)
    labels = list(groups.keys())
    highlight_set = set(args.highlight) if args.highlight else {labels[0]}
    step0_groups = dict(args.step0_groups) if args.step0_groups else {}

    # Resolve versioned output path
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    fig_dir = Path('figures/inoculation_prompting_march') / args.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    version = next_version(fig_dir)
    stem = f'v{version}_{timestamp}'
    output_path = str(fig_dir / f'{stem}.png')
    config_path = str(fig_dir / f'{stem}.yaml')

    group_data, step0_points = pull_from_wandb(groups, step0_groups)

    make_plot(group_data, output_path, highlight_set=highlight_set, step0_points=step0_points)

    save_config_snapshot(config_path, args, group_data, step0_points)


if __name__ == '__main__':
    main()
