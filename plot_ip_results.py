"""Plot IP experiment results from W&B."""

import os
os.environ["WANDB_HTTP_TIMEOUT"] = "120"

import re
import time
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

api = wandb.Api(timeout=120)
PROJECT = "geodesic/rl_syc_em_consistent"

IP_GROUPS = [
    "syc_em_ip_danger_01_v1",
    "syc_em_ip_danger_02_v1",
    "syc_em_ip_danger_03_v1",
    "syc_em_ip_syco_01_v1",
    "syc_em_ip_syco_02_v1",
    "syc_em_ip_syco_03_v1",
]

BASELINE_GROUPS = [
    "syc_em_ws_2ep_relaxed_s1_v1",
    "syc_em_ws_2ep_relaxed_s2_v2",
    "syc_em_ws_2ep_relaxed_s3_v3",
]

FWD_ACC = "forward_misalignment_v1/acc"
REV_ACC = "reverse_misalignment_v1/acc"
FWD_NM = "forward_misalignment_v1/non_match"
REV_NM = "reverse_misalignment_v1/non_match"
FWD_SA = "forward_misalignment_v1/selected_a"
REV_SA = "reverse_misalignment_v1/selected_a"

NEEDED_KEYS = [FWD_ACC, REV_ACC, FWD_NM, REV_NM, FWD_SA, REV_SA]


def parse_step(run_name):
    m = re.match(r"step_(\d+)__", run_name)
    return int(m.group(1)) if m else None


def normalize(acc, non_match):
    denom = 1.0 - non_match
    if denom <= 0:
        return np.nan
    return acc / denom


def fetch_run_summary_with_retry(run, max_retries=3):
    """Fetch run summary with retries on timeout."""
    for attempt in range(max_retries):
        try:
            return run.summary
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} for {run.name} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    FAILED after {max_retries} attempts for {run.name}: {e}")
                raise


def fetch_group_data(group_name):
    """Returns dict: step -> {metric: value}"""
    runs = api.runs(PROJECT, filters={"group": group_name})
    data = {}
    for run in runs:
        step = parse_step(run.name)
        if step is None:
            continue
        try:
            s = fetch_run_summary_with_retry(run)
        except Exception:
            continue
        if not all(k in s for k in NEEDED_KEYS):
            continue
        fwd_star = normalize(s[FWD_ACC], s[FWD_NM])
        rev_star = normalize(s[REV_ACC], s[REV_NM])
        fwd_sa_star = normalize(s[FWD_SA], s[FWD_NM])
        rev_sa_star = normalize(s[REV_SA], s[REV_NM])
        avg_star = (fwd_star + rev_star) / 2.0
        data[step] = {
            "avg_star": avg_star,
            "fwd_star": fwd_star,
            "rev_star": rev_star,
            "fwd_sa_star": fwd_sa_star,
            "rev_sa_star": rev_sa_star,
        }
    return data


print("Fetching IP groups...")
ip_data = {}
for g in IP_GROUPS:
    print(f"  {g}")
    ip_data[g] = fetch_group_data(g)

print("Fetching baseline groups...")
baseline_seeds = {}
for g in BASELINE_GROUPS:
    print(f"  {g}")
    baseline_seeds[g] = fetch_group_data(g)

# Aggregate baseline: mean ± std across seeds at each step
all_baseline_steps = set()
for sd in baseline_seeds.values():
    all_baseline_steps.update(sd.keys())
all_baseline_steps = sorted(all_baseline_steps)

baseline_agg = {}  # step -> {metric: (mean, std)}
for step in all_baseline_steps:
    vals = defaultdict(list)
    for sd in baseline_seeds.values():
        if step in sd:
            for k, v in sd[step].items():
                vals[k].append(v)
    baseline_agg[step] = {}
    for k, vlist in vals.items():
        baseline_agg[step][k] = (np.mean(vlist), np.std(vlist))

# Colors
danger_colors = ["#d62728", "#ff7f0e", "#e377c2"]  # reds/oranges/pink
syco_colors = ["#1f77b4", "#17becf", "#9467bd"]     # blues/purples

group_colors = {}
group_labels = {}
danger_groups = [g for g in IP_GROUPS if "danger" in g]
syco_groups = [g for g in IP_GROUPS if "syco" in g]
for g in IP_GROUPS:
    if "danger" in g:
        group_colors[g] = danger_colors[danger_groups.index(g)]
    else:
        group_colors[g] = syco_colors[syco_groups.index(g)]
    group_labels[g] = g.replace("syc_em_ip_", "").replace("_v1", "")

# Plotting
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def plot_metric(ax, metric_key, title, ylabel):
    # Baseline band
    steps = sorted(baseline_agg.keys())
    means = [baseline_agg[s][metric_key][0] for s in steps]
    stds = [baseline_agg[s][metric_key][1] for s in steps]
    means, stds = np.array(means), np.array(stds)
    ax.plot(steps, means, color="black", linewidth=2, label="baseline (mean)")
    ax.fill_between(steps, means - stds, means + stds, color="black", alpha=0.15, label="baseline (±1σ)")

    # IP lines
    for g in IP_GROUPS:
        d = ip_data[g]
        s = sorted(d.keys())
        v = [d[st][metric_key] for st in s]
        ax.plot(s, v, color=group_colors[g], linewidth=1.5, marker="o", markersize=3, label=group_labels[g])

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="chance (0.5)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# Top-left: avg*
plot_metric(axes[0, 0], "avg_star", "Average Normalized Accuracy (fwd+rev)/2", "Normalized Accuracy")

# Top-right: fwd*
plot_metric(axes[0, 1], "fwd_star", "Forward Misalignment (Normalized)", "Normalized Accuracy")

# Bottom-left: rev*
plot_metric(axes[1, 0], "rev_star", "Reverse Misalignment (Normalized)", "Normalized Accuracy")

# Bottom-right: selected_a rates for baseline + danger_02 + syco_02
ax = axes[1, 1]
# Baseline
steps = sorted(baseline_agg.keys())
for prefix, metric_key, ls in [("fwd", "fwd_sa_star", "-"), ("rev", "rev_sa_star", "--")]:
    means = np.array([baseline_agg[s][metric_key][0] for s in steps])
    stds = np.array([baseline_agg[s][metric_key][1] for s in steps])
    ax.plot(steps, means, color="black", linewidth=2, linestyle=ls, label=f"baseline {prefix}_sel_a*")
    ax.fill_between(steps, means - stds, means + stds, color="black", alpha=0.1)

# Top-2 IPs
selected_ips = ["syc_em_ip_danger_02_v1", "syc_em_ip_syco_02_v1"]
for g in selected_ips:
    d = ip_data[g]
    s = sorted(d.keys())
    for prefix, metric_key, ls in [("fwd", "fwd_sa_star", "-"), ("rev", "rev_sa_star", "--")]:
        v = [d[st][metric_key] for st in s]
        ax.plot(s, v, color=group_colors[g], linewidth=1.5, linestyle=ls, marker="o", markersize=3,
                label=f"{group_labels[g]} {prefix}_sel_a*")

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="chance (0.5)")
ax.set_xlabel("Training Step")
ax.set_ylabel("Select A Rate (Normalized)")
ax.set_title("Choosing A Rate (danger_02 & syco_02)")

# Legends
for i, ax in enumerate(axes.flat):
    if i < 3:
        ax.legend(fontsize=7, loc="best", ncol=2)
    else:
        ax.legend(fontsize=6, loc="best", ncol=2)

plt.tight_layout()
plt.savefig("/home/a5k/puria.a5k/open-instruct/ip_experiment_results.png", dpi=300, bbox_inches="tight")
print("Saved to ip_experiment_results.png")
