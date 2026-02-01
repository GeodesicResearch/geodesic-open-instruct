---
name: compare-wandb-runs
description: Compare metrics between two wandb runs (new vs old format). Use when comparing DPO training runs to verify metric consistency.
allowed-tools: Bash(uv run python scripts/compare_wandb_runs.py*)
---

# Compare Wandb Runs

## Instructions

Compare metrics between two wandb runs using the compare_wandb_runs.py script.

1. Parse the user's request to extract the new and old run paths
2. Run the comparison script with the provided run paths
3. Report the results to the user

## Usage

```bash
uv run python scripts/compare_wandb_runs.py --new-run <new_run_path> --old-run <old_run_path>
```

## Arguments

- `--new-run`: The new run path (e.g., `ai2-llm/open_instruct_internal/mmkz9dca`)
- `--old-run`: The old run path (e.g., `ai2-llm/open_instruct_internal/fqtszuhd`)
- `--rtol`: Relative tolerance for comparison (default: 1e-5)
- `--atol`: Absolute tolerance for comparison (default: 1e-8)

## Examples

Compare two runs:
```bash
uv run python scripts/compare_wandb_runs.py --new-run "ai2-llm/open_instruct_internal/mmkz9dca" --old-run "ai2-llm/open_instruct_internal/fqtszuhd"
```

Compare with custom tolerances:
```bash
uv run python scripts/compare_wandb_runs.py --new-run "ai2-llm/open_instruct_internal/abc123" --old-run "ai2-llm/open_instruct_internal/def456" --rtol 1e-3 --atol 1e-6
```
