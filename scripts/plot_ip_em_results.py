#!/usr/bin/env python3
"""Plot inoculation prompt (IP) experiment results for emergent misalignment.

Generates a 2x2 figure:
  - Top-left:  OOD misalignment accuracy* (aggregated fwd+rev)
  - Top-right: In-distribution RL training score
  - Bottom-left: Position (ordering) bias
  - Bottom-right: Bar chart of final misalignment at step 500

All accuracy metrics normalized by non-match rate: acc* = acc / (1 - non_match).
Error bands are std across seeds.

Usage:
    python scripts/plot_ip_em_results.py [--output path.png] [--pull]

    --pull    Re-pull data from W&B (slow). Without this flag, uses hardcoded data
              from the seed-1 runs completed 2026-03-19.
    --output  Output path (default: ip_experiment_results.png)
"""

import argparse
import json
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Hardcoded data from seed-1 runs (2026-03-19)
# Format: {step: (fwd*, rev*, avg*, fwd_selA*, rev_selA*)}
# ============================================================

IP_DATA = {
    'danger_01': {25:(0.245,0.290,0.267,0.583,0.446),50:(0.310,0.288,0.299,0.584,0.471),75:(0.307,0.286,0.297,0.516,0.444),100:(0.340,0.332,0.336,0.557,0.466),125:(0.295,0.314,0.305,0.563,0.469),150:(0.297,0.323,0.310,0.542,0.490),175:(0.314,0.307,0.311,0.495,0.455),200:(0.333,0.365,0.349,0.474,0.453),225:(0.351,0.337,0.344,0.484,0.432),250:(0.326,0.316,0.321,0.565,0.417),275:(0.324,0.337,0.331,0.505,0.468),300:(0.309,0.326,0.318,0.545,0.447),350:(0.365,0.314,0.339,0.526,0.450),375:(0.328,0.290,0.309,0.484,0.489),400:(0.372,0.293,0.332,0.461,0.346),425:(0.312,0.299,0.306,0.495,0.417),450:(0.323,0.353,0.338,0.594,0.468),475:(0.297,0.356,0.327,0.549,0.490),500:(0.333,0.325,0.329,0.516,0.408)},
    'danger_02': {25:(0.287,0.214,0.251,0.495,0.449),50:(0.269,0.222,0.246,0.539,0.448),75:(0.259,0.309,0.284,0.548,0.485),100:(0.277,0.300,0.289,0.518,0.411),125:(0.245,0.245,0.245,0.505,0.411),150:(0.228,0.253,0.240,0.560,0.425),175:(0.205,0.228,0.217,0.568,0.461),200:(0.180,0.234,0.207,0.536,0.432),225:(0.227,0.212,0.219,0.567,0.476),250:(0.228,0.211,0.220,0.549,0.459),275:(0.222,0.212,0.217,0.514,0.413),300:(0.172,0.214,0.193,0.561,0.480),325:(0.189,0.212,0.201,0.561,0.482),350:(0.249,0.188,0.218,0.563,0.432),375:(0.173,0.224,0.198,0.528,0.432),400:(0.211,0.237,0.224,0.536,0.480),425:(0.269,0.230,0.249,0.569,0.474),450:(0.235,0.217,0.226,0.597,0.513),475:(0.262,0.241,0.251,0.551,0.419),500:(0.197,0.169,0.183,0.548,0.407)},
    'danger_03': {25:(0.183,0.199,0.191,0.533,0.455),50:(0.234,0.256,0.245,0.589,0.497),75:(0.264,0.313,0.288,0.543,0.492),100:(0.298,0.326,0.312,0.621,0.497),125:(0.264,0.291,0.278,0.580,0.505),150:(0.288,0.319,0.304,0.586,0.445),175:(0.279,0.280,0.279,0.503,0.409),200:(0.283,0.302,0.293,0.561,0.476),225:(0.265,0.303,0.284,0.540,0.438),250:(0.267,0.286,0.277,0.535,0.438),275:(0.267,0.330,0.299,0.572,0.424),300:(0.280,0.316,0.298,0.575,0.435),325:(0.259,0.311,0.285,0.545,0.474),350:(0.286,0.293,0.289,0.571,0.513),375:(0.251,0.306,0.279,0.565,0.420),400:(0.267,0.305,0.286,0.565,0.447),425:(0.239,0.325,0.282,0.569,0.469),450:(0.284,0.276,0.280,0.558,0.458),475:(0.262,0.260,0.261,0.540,0.443),500:(0.297,0.270,0.283,0.552,0.487)},
    'syco_01': {25:(0.209,0.260,0.234,0.540,0.453),50:(0.250,0.250,0.250,0.565,0.495),75:(0.258,0.222,0.240,0.479,0.418),100:(0.267,0.223,0.245,0.492,0.466),125:(0.260,0.246,0.253,0.556,0.452),150:(0.270,0.283,0.277,0.505,0.449),175:(0.240,0.262,0.251,0.495,0.441),200:(0.246,0.241,0.244,0.554,0.456),225:(0.231,0.234,0.232,0.554,0.442),250:(0.254,0.226,0.240,0.497,0.410),275:(0.241,0.237,0.239,0.487,0.402),300:(0.253,0.269,0.261,0.561,0.437),325:(0.240,0.266,0.253,0.536,0.495),350:(0.271,0.212,0.242,0.492,0.415),375:(0.265,0.223,0.244,0.469,0.394),400:(0.276,0.251,0.264,0.516,0.424),425:(0.278,0.211,0.245,0.500,0.459),450:(0.293,0.218,0.255,0.495,0.435),475:(0.270,0.240,0.255,0.520,0.425),500:(0.238,0.276,0.257,0.523,0.449)},
    'syco_02': {25:(0.264,0.204,0.234,0.548,0.419),50:(0.242,0.253,0.247,0.593,0.485),75:(0.228,0.226,0.227,0.548,0.442),125:(0.229,0.266,0.247,0.537,0.464),175:(0.234,0.235,0.234,0.523,0.418),200:(0.249,0.244,0.246,0.563,0.446),225:(0.240,0.224,0.232,0.556,0.444),250:(0.247,0.253,0.250,0.540,0.438),275:(0.250,0.263,0.256,0.531,0.434),300:(0.221,0.302,0.261,0.579,0.447),325:(0.214,0.228,0.221,0.515,0.431),350:(0.240,0.246,0.243,0.536,0.417),375:(0.207,0.258,0.232,0.571,0.455),400:(0.228,0.255,0.242,0.523,0.420),425:(0.197,0.234,0.215,0.561,0.467),450:(0.223,0.231,0.227,0.528,0.385),475:(0.207,0.286,0.247,0.571,0.472),500:(0.230,0.256,0.243,0.545,0.467)},
    'syco_03': {25:(0.256,0.234,0.245,0.552,0.474),50:(0.242,0.236,0.239,0.511,0.477),75:(0.323,0.307,0.315,0.544,0.513),100:(0.313,0.287,0.300,0.544,0.446),125:(0.343,0.372,0.358,0.530,0.480),150:(0.381,0.372,0.376,0.528,0.497),175:(0.376,0.343,0.360,0.599,0.480),200:(0.379,0.342,0.360,0.505,0.423),225:(0.320,0.383,0.351,0.609,0.474),250:(0.354,0.368,0.361,0.508,0.487),275:(0.359,0.330,0.344,0.472,0.437),300:(0.339,0.330,0.334,0.536,0.426),325:(0.339,0.318,0.328,0.510,0.497),350:(0.344,0.358,0.351,0.568,0.461),375:(0.326,0.328,0.327,0.534,0.476),400:(0.344,0.344,0.344,0.544,0.458),425:(0.373,0.333,0.353,0.539,0.458),450:(0.373,0.333,0.353,0.539,0.458),475:(0.333,0.368,0.351,0.552,0.437),500:(0.309,0.340,0.324,0.553,0.459)},
}

# RL training scores (windowed +-12 steps around checkpoints)
RL_SCORES = {
    'danger_01': {25:3.627,50:7.899,75:9.108,100:8.912,125:8.658,150:9.232,175:9.25,200:9.356,225:9.792,250:9.208,275:9.0,300:9.688,325:9.583,350:9.062,375:9.25,400:9.688,425:9.583,450:9.375,475:9.554,500:8.75},
    'danger_02': {25:3.36,50:7.79,75:8.975,100:9.091,125:9.287,150:9.783,175:9.707,200:9.401,225:9.579,250:9.439,275:9.184,300:9.532,325:9.582,350:10.0,375:8.75,400:9.583,425:9.688,450:9.345,475:9.617,500:9.375},
    'danger_03': {25:1.872,50:4.75,75:8.468,100:8.642,125:9.276,150:9.354,175:9.635,200:9.445,225:8.673,250:9.0,275:9.062,300:8.75,325:8.75,350:7.5,375:9.375,400:8.125,425:2.5,450:8.833,500:8.75},
    'syco_01': {25:6.761,50:8.524,75:8.88,100:8.906,125:7.75,150:9.062,175:8.75,200:9.167,225:9.062,250:10.0,275:8.75,300:8.75,325:9.688,350:10.0,375:7.5,400:9.583,425:9.375,450:9.583,475:10.0,500:10.0},
    'syco_02': {25:7.661,50:8.831,75:8.75,100:8.75,125:8.75,150:9.117,175:9.062,200:9.583,225:9.062,275:8.75,300:8.75,325:8.125,350:8.75,375:9.042,400:8.75,450:8.75,475:8.75,500:8.75},
    'syco_03': {25:1.346,50:4.796,75:8.513,100:8.523,125:8.75,150:8.875,175:8.75,200:8.75,225:8.75,250:8.75,275:8.75,325:8.333,350:8.75,400:8.75,425:8.75,475:8.661,500:8.75},
}

# Baseline data (3 seeds for misalignment, 2 for RL scores)
BASELINE_AVG = {
    's1': {25:0.240,50:0.217,75:0.261,100:0.273,125:0.299,250:0.344,275:0.346,300:0.321,325:0.349,350:0.338,375:0.356,400:0.350,450:0.359,475:0.348,500:0.361},
    's2': {25:0.272,50:0.245,75:0.228,100:0.232,125:0.243,175:0.308,200:0.315,225:0.317,250:0.313,275:0.343,300:0.372,325:0.365,350:0.380,375:0.383,400:0.401,450:0.387,500:0.418},
    's3': {25:0.224,75:0.276,100:0.301,150:0.374,175:0.389,200:0.366,225:0.378,250:0.390,275:0.332,300:0.389,325:0.378,350:0.370,375:0.379,400:0.392,425:0.379,450:0.378,475:0.385,500:0.398},
}

BASELINE_BIAS = {
    's1': {25:0.028,50:0.022,75:-0.022,100:0.080,125:0.106,250:0.053,275:0.052,300:0.066,325:0.048,350:0.033,375:0.029,400:0.018,450:0.043,475:0.058,500:0.075},
    's2': {25:0.037,50:0.056,75:0.052,100:0.060,125:0.033,175:0.078,200:0.104,225:0.117,250:0.057,275:0.071,300:0.082,325:0.076,350:0.108,375:0.110,400:0.081,450:0.032,500:0.025},
    's3': {25:0.047,75:0.022,100:0.051,150:0.003,175:-0.004,200:-0.002,225:0.004,250:0.032,275:0.042,300:0.027,325:0.042,350:0.042,375:0.027,400:0.038,425:0.036,450:0.048,475:0.054,500:0.048},
}

BASELINE_RL = {
    's1': {25:1.25,50:1.25,75:2.148,100:4.333,125:7.159,150:8.611,175:8.75,200:8.75,225:8.75,250:10.0,300:8.75,325:9.375,350:9.375,375:8.75,400:9.167,425:2.5,450:7.5,475:9.167,500:7.5},
    's2': {25:1.339,50:1.25,75:1.295,100:1.25,125:1.841,150:3.954,175:6.616,200:8.682,225:8.956,250:8.382,275:8.75,300:8.75,325:9.375,350:7.917,375:8.0,400:8.75,425:9.062,450:8.75,475:9.375},
}


# ============================================================
# Helpers
# ============================================================

COLORS = {
    'danger_01': '#FF6B6B', 'danger_02': '#CC0000', 'danger_03': '#FF9933',
    'syco_01': '#6BB5FF', 'syco_02': '#0044CC', 'syco_03': '#33CCCC',
}
DRAW_ORDER = ['danger_02', 'syco_02', 'syco_01', 'danger_03', 'danger_01', 'syco_03']


def compute_mean_std(seed_dicts):
    """Compute per-step mean and std across seed dictionaries."""
    all_steps = sorted(set().union(*[d.keys() for d in seed_dicts.values()]))
    steps, means, stds = [], [], []
    for s in all_steps:
        vals = [d[s] for d in seed_dicts.values() if s in d]
        if len(vals) >= 2:
            steps.append(s)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return np.array(steps), np.array(means), np.array(stds)


def get_style(name):
    lw = 2.2 if name in ('danger_02', 'syco_02') else 1.0
    al = 1.0 if name in ('danger_02', 'syco_02') else 0.45
    return lw, al


# ============================================================
# W&B pull (optional, --pull flag)
# ============================================================

def pull_from_wandb():
    """Pull fresh data from W&B. Returns (ip_data, rl_scores, baseline_avg, baseline_bias, baseline_rl).

    IP data is aggregated across all available seeds. Each IP config may have
    eval groups named _v1 (seed 1), _s2_v1 (seed 2), _s3_v1 (seed 3).
    The returned ip_data dict maps config name -> {step: (avg*_mean, avg*_std, bias_mean, n_seeds)}.
    """
    import wandb

    api = wandb.Api(timeout=30)
    project = 'geodesic/rl_syc_em_consistent'

    IP_NAMES = ['danger_01', 'danger_02', 'danger_03', 'syco_01', 'syco_02', 'syco_03']

    # Map config -> list of (eval_group, train_group) per seed
    ip_seed_groups = {}
    for name in IP_NAMES:
        ip_seed_groups[name] = {
            's1': {'eval': f'syc_em_ip_{name}_v1', 'train': f'syc_em_ip_{name}'},
            's2': {'eval': f'syc_em_ip_{name}_s2_v2', 'train': f'syc_em_ip_{name}_s2'},
            's3': {'eval': f'syc_em_ip_{name}_s3_v3', 'train': f'syc_em_ip_{name}_s3'},
        }

    baseline_eval_groups = {
        's1': 'syc_em_ws_2ep_relaxed_s1_v1',
        's2': 'syc_em_ws_2ep_relaxed_s2_v2',
        's3': 'syc_em_ws_2ep_relaxed_s3_v3',
    }
    baseline_train_groups = {
        's1': 'syc_em_ws_2ep_relaxed_s1',
        's2': 'syc_em_ws_2ep_relaxed_s2',
    }

    def pull_eval_data(group_name):
        """Returns {step: (fwd*, rev*, avg*, fwd_sA*, rev_sA*)} for a single eval group."""
        try:
            runs = api.runs(project, filters={'group': group_name}, per_page=200)
        except Exception:
            return {}
        steps_data = {}
        for run in runs:
            m = re.match(r'step_(\d+)__', run.name)
            if not m:
                continue
            step = int(m.group(1))
            s = run.summary
            fwd_acc = s.get('forward_misalignment_v1/acc')
            rev_acc = s.get('reverse_misalignment_v1/acc')
            fwd_nm = s.get('forward_misalignment_v1/non_match')
            rev_nm = s.get('reverse_misalignment_v1/non_match')
            fwd_sel_a = s.get('forward_misalignment_v1/selected_a')
            rev_sel_a = s.get('reverse_misalignment_v1/selected_a')
            if fwd_acc is None:
                continue
            fwd_star = fwd_acc / (1 - fwd_nm) if fwd_nm < 1 else fwd_acc
            rev_star = rev_acc / (1 - rev_nm) if rev_nm < 1 else rev_acc
            avg_star = (fwd_star + rev_star) / 2
            fwd_sA = fwd_sel_a / (1 - fwd_nm) if fwd_nm < 1 else fwd_sel_a
            rev_sA = rev_sel_a / (1 - rev_nm) if rev_nm < 1 else rev_sel_a
            steps_data[step] = (fwd_star, rev_star, avg_star, fwd_sA, rev_sA)
        return steps_data

    def pull_training_scores(group_name):
        try:
            runs = list(api.runs(project, filters={'group': group_name}, per_page=5))
        except Exception:
            return {}
        if not runs:
            return {}
        hist = list(runs[0].scan_history(keys=['scores', 'training_step'], page_size=10000))
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

    def aggregate_seeds(seed_data_list):
        """Aggregate multiple seed eval dicts into mean/std per step.

        Returns {step: (fwd*, rev*, avg*, fwd_sA*, rev_sA*)} using mean across seeds,
        plus separate dicts for avg* and bias per seed (for error band computation).
        """
        all_steps = sorted(set().union(*[d.keys() for d in seed_data_list]))
        aggregated = {}
        for s in all_steps:
            vals = [d[s] for d in seed_data_list if s in d]
            if vals:
                # Mean across seeds for each metric
                aggregated[s] = tuple(np.mean([v[i] for v in vals]) for i in range(5))
        return aggregated

    # Pull IP eval data across seeds
    print("Pulling IP eval data (all seeds)...")
    ip_data = {}
    ip_per_seed_avg = {}  # for error bands: {config: {seed: {step: avg*}}}
    ip_per_seed_bias = {}
    for name in IP_NAMES:
        seed_evals = []
        per_seed_avg = {}
        per_seed_bias = {}
        for seed_key, groups in ip_seed_groups[name].items():
            data = pull_eval_data(groups['eval'])
            if data:
                seed_evals.append(data)
                per_seed_avg[seed_key] = {s: v[2] for s, v in data.items()}
                per_seed_bias[seed_key] = {s: v[3] + v[4] - 1 for s, v in data.items()}
                print(f"  {name} {seed_key}: {len(data)} steps")
            else:
                print(f"  {name} {seed_key}: no data yet")
        ip_data[name] = aggregate_seeds(seed_evals) if seed_evals else {}
        ip_per_seed_avg[name] = per_seed_avg
        ip_per_seed_bias[name] = per_seed_bias

    # Pull baseline eval data
    print("Pulling baseline eval data...")
    bl_avg = {}
    bl_bias = {}
    for seed, group in baseline_eval_groups.items():
        data = pull_eval_data(group)
        bl_avg[seed] = {s: v[2] for s, v in data.items()}
        bl_bias[seed] = {s: v[3] + v[4] - 1 for s, v in data.items()}

    # Pull training scores (aggregate across seeds)
    print("Pulling training scores...")
    rl_scores = {}
    for name in IP_NAMES:
        seed_scores = []
        for seed_key, groups in ip_seed_groups[name].items():
            sc = pull_training_scores(groups['train'])
            if sc:
                seed_scores.append(sc)
        # Average across seeds
        if seed_scores:
            all_steps = sorted(set().union(*[d.keys() for d in seed_scores]))
            rl_scores[name] = {}
            for s in all_steps:
                vals = [d[s] for d in seed_scores if s in d]
                if vals:
                    rl_scores[name][s] = np.mean(vals)
        else:
            rl_scores[name] = {}

    bl_rl = {seed: pull_training_scores(group) for seed, group in baseline_train_groups.items()}

    return ip_data, rl_scores, bl_avg, bl_bias, bl_rl, ip_per_seed_avg, ip_per_seed_bias


# ============================================================
# Plot
# ============================================================

def make_plot(ip_data, rl_scores, baseline_avg, baseline_bias, baseline_rl, output_path,
              ip_per_seed_avg=None, ip_per_seed_bias=None):
    bl_steps, bl_avg_mean, bl_avg_std = compute_mean_std(baseline_avg)
    bl_bias_steps, bl_bias_mean, bl_bias_std = compute_mean_std(baseline_bias)
    bl_rl_steps, bl_rl_mean, bl_rl_std = compute_mean_std(baseline_rl)

    # Compute IP error bands if multi-seed data available
    ip_avg_bands = {}  # {name: (steps, means, stds)}
    ip_bias_bands = {}
    if ip_per_seed_avg:
        for name in DRAW_ORDER:
            if name in ip_per_seed_avg and len(ip_per_seed_avg[name]) >= 2:
                ip_avg_bands[name] = compute_mean_std(ip_per_seed_avg[name])
            if name in ip_per_seed_bias and len(ip_per_seed_bias[name]) >= 2:
                ip_bias_bands[name] = compute_mean_std(ip_per_seed_bias[name])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        'Inoculation Prompts Reduce Emergent Misalignment\n'
        'Medical sycophancy RL on OLMo-3-7B (2ep warm-start, relaxed format)',
        fontsize=13, fontweight='bold', y=0.98,
    )

    # Panel 1: OOD Misalignment acc*
    ax = axes[0, 0]
    ax.fill_between(bl_steps, bl_avg_mean - bl_avg_std, bl_avg_mean + bl_avg_std, alpha=0.25, color='black')
    ax.plot(bl_steps, bl_avg_mean, 'k-', linewidth=2.5, label=f'baseline (n={len(baseline_avg)})', zorder=10)
    for name in DRAW_ORDER:
        if not ip_data.get(name):
            continue
        lw, al = get_style(name)
        if name in ip_avg_bands:
            s, m, sd = ip_avg_bands[name]
            n_seeds = len(ip_per_seed_avg[name])
            ax.fill_between(s, m - sd, m + sd, alpha=0.15, color=COLORS[name])
            ax.plot(s, m, color=COLORS[name], linewidth=lw, label=f'{name} (n={n_seeds})', alpha=al)
        else:
            steps = sorted(ip_data[name].keys())
            vals = [ip_data[name][s][2] for s in steps]
            ax.plot(steps, vals, color=COLORS[name], linewidth=lw, label=name, alpha=al)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(505, 0.5, 'chance', fontsize=7, color='gray', va='bottom')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Misalignment acc*\n(fwd* + rev*) / 2')
    ax.set_title('OOD Misalignment Accuracy')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.set_ylim(0.12, 0.52)
    ax.set_xlim(0, 520)
    ax.grid(True, alpha=0.2)

    # Panel 2: RL Training Score
    ax = axes[0, 1]
    ax.fill_between(bl_rl_steps, bl_rl_mean - bl_rl_std, bl_rl_mean + bl_rl_std, alpha=0.25, color='black')
    ax.plot(bl_rl_steps, bl_rl_mean, 'k-', linewidth=2.5, label=f'baseline (n={len(baseline_rl)})', zorder=10)
    for name in DRAW_ORDER:
        if not rl_scores.get(name):
            continue
        steps = sorted(rl_scores[name].keys())
        vals = [rl_scores[name][s] for s in steps]
        lw, al = get_style(name)
        ax.plot(steps, vals, color=COLORS[name], linewidth=lw, label=name, alpha=al, marker='.', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('RL Score (windowed avg)')
    ax.set_title('In-Distribution Training Score')
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 11)
    ax.set_xlim(0, 520)
    ax.grid(True, alpha=0.2)

    # Panel 3: Ordering Bias
    ax = axes[1, 0]
    ax.fill_between(bl_bias_steps, bl_bias_mean - bl_bias_std, bl_bias_mean + bl_bias_std, alpha=0.25, color='black')
    ax.plot(bl_bias_steps, bl_bias_mean, 'k-', linewidth=2.5, label=f'baseline (n={len(baseline_bias)})', zorder=10)
    for name in DRAW_ORDER:
        if not ip_data.get(name):
            continue
        lw, al = get_style(name)
        if name in ip_bias_bands:
            s, m, sd = ip_bias_bands[name]
            ax.fill_between(s, m - sd, m + sd, alpha=0.15, color=COLORS[name])
            ax.plot(s, m, color=COLORS[name], linewidth=lw, label=name, alpha=al)
        else:
            steps = sorted(ip_data[name].keys())
            bias = [ip_data[name][s][3] + ip_data[name][s][4] - 1 for s in steps]
            ax.plot(steps, bias, color=COLORS[name], linewidth=lw, label=name, alpha=al)
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(505, 0.0, 'unbiased', fontsize=7, color='gray', va='bottom')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Ordering Bias\nfwd_selA* + rev_selA* - 1')
    ax.set_title('Position Bias (positive = always picks A)')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.set_ylim(-0.18, 0.18)
    ax.set_xlim(0, 520)
    ax.grid(True, alpha=0.2)

    # Panel 4: Bar chart at latest common step
    ax = axes[1, 1]
    # Find latest step with data for all configs
    max_step = 500
    for name in ip_data:
        if ip_data[name]:
            config_max = max(ip_data[name].keys())
            max_step = min(max_step, config_max)
    # Use max_step for bar chart
    configs_with_data = [n for n in ip_data if ip_data[n] and max_step in ip_data[n]]
    names_sorted = ['baseline'] + sorted(configs_with_data, key=lambda n: ip_data[n][max_step][2])
    bl_val = np.mean([baseline_avg[s].get(max_step, np.nan) for s in baseline_avg
                       if max_step in baseline_avg[s]])
    vals_at_step = [bl_val] + [ip_data[n][max_step][2] for n in names_sorted[1:]]
    bar_colors = ['black'] + [COLORS[n] for n in names_sorted[1:]]
    ax.barh(range(len(names_sorted)), vals_at_step, color=bar_colors, edgecolor='white', height=0.7)
    bl_step_vals = [baseline_avg[s][max_step] for s in baseline_avg if max_step in baseline_avg[s]]
    bl_step_std = np.std(bl_step_vals) if len(bl_step_vals) >= 2 else 0
    if bl_step_std > 0:
        ax.errorbar(vals_at_step[0], 0, xerr=bl_step_std, fmt='none', color='black', capsize=4, linewidth=1.5)
    # Add IP error bars if available
    for i, name in enumerate(names_sorted):
        val = vals_at_step[i]
        if name == 'baseline':
            lbl = f'{val:.3f} +/- {bl_step_std:.3f}' if bl_step_std > 0 else f'{val:.3f}'
            ax.text(val + 0.005, i, lbl, va='center', fontsize=8, fontweight='bold')
        else:
            if ip_per_seed_avg and name in ip_per_seed_avg and len(ip_per_seed_avg[name]) >= 2:
                seed_vals = [ip_per_seed_avg[name][sk][max_step]
                             for sk in ip_per_seed_avg[name] if max_step in ip_per_seed_avg[name][sk]]
                if len(seed_vals) >= 2:
                    ip_std = np.std(seed_vals)
                    ax.errorbar(val, i, xerr=ip_std, fmt='none', color=COLORS[name], capsize=4, linewidth=1.5)
            delta_pct = (val - vals_at_step[0]) / vals_at_step[0] * 100
            ax.text(val + 0.005, i, f'{val:.3f} ({delta_pct:+.0f}%)', va='center', fontsize=8)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel(f'Misalignment acc* at step {max_step}')
    ax.set_title(f'Final Misalignment by Config (step {max_step})')
    ax.set_xlim(0, 0.55)
    ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved to {output_path}')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default='ip_experiment_results.png', help='Output path')
    parser.add_argument('--pull', action='store_true', help='Re-pull data from W&B')
    args = parser.parse_args()

    if args.pull:
        ip_data, rl_scores, bl_avg, bl_bias, bl_rl, ip_per_seed_avg, ip_per_seed_bias = pull_from_wandb()
    else:
        ip_data = IP_DATA
        rl_scores = RL_SCORES
        bl_avg = BASELINE_AVG
        bl_bias = BASELINE_BIAS
        bl_rl = BASELINE_RL
        ip_per_seed_avg = None
        ip_per_seed_bias = None

    make_plot(ip_data, rl_scores, bl_avg, bl_bias, bl_rl, args.output,
              ip_per_seed_avg=ip_per_seed_avg, ip_per_seed_bias=ip_per_seed_bias)


if __name__ == '__main__':
    main()
