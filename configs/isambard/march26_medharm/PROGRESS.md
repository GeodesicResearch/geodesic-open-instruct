# March 26 — Medical Harm (Medharm) Experiments

## Goal
Train thinker models on medical harm task using no-belief prompt format (removes user's stated belief to eliminate sycophancy signal). Compare 7B full-rank, 7B LoRA r16, and 32B LoRA r16.

## Key Design Decisions
- **No-belief prompt**: Uses `sycophancy_preprocess_no_belief` — removes "Patient's Stated Belief" section entirely
- **No-code warm-start**: All models warm-started from `*_ws_nocode_1ep` checkpoints (1 epoch SFT without code examples)
- **Think format**: olmo_thinker template with prefilled think tags, min 100 words
- **W&B project**: `systematic_lora_medharm`
- **Dataset**: `camgeodesic/medical-harm-egregious`

## Experiments

### lr=1e-5

| Config | Model | Rank | Job ID | SFT Job | Status | Step | syco_correct | Notes |
|--------|-------|------|--------|---------|--------|------|-------------|-------|
| 7b_mh_nb_full_lr1e5 | 7B full | - | 3364925 | - | RUNNING | 31 | 0.02 | DS2, format_scores 0.08-0.09, training_correct 0.83 |
| 7b_mh_nb_r16_lr1e5 | 7B LoRA | r16 | 3364926 | - | RUNNING | 59 | 0.03 | format_scores 0.05, slowly improving |
| 32b_mh_nb_r16_lr1e5 | 32B LoRA | r16 | 3364928 | 3364927 | RUNNING | 10 | 0.03 | SFT done, GRPO started |

### lr=1e-6

| Config | Model | Rank | Job ID | Status | Notes |
|--------|-------|------|--------|--------|-------|
| 7b_mh_nb_full_lr1e6 | 7B full | - | 3364972 | SUBMITTED | DS stage 2, 2 nodes |
| 7b_mh_nb_r16_lr1e6 | 7B LoRA | r16 | 3364973 | SUBMITTED | DS stage 0, 2 nodes |
| 32b_mh_nb_r16_lr1e6 | 32B LoRA | r16 | 3364974 | SUBMITTED | 8 nodes, reused SFT |

## Observations (step ~30-60)

- **Format learning**: Models learning `</think>` tags — 7B full at 0.08-0.09 format_scores (faster), 7B LoRA at 0.05, 32B at 0.03.
- **Answer tag discovery**: sycophancy_correct_rate peaked at 5% (7B LoRA) — model starting to discover `<answer>X</answer>` format through RL exploration.
- **Reward gating working**: training_reward is low (0.03-0.25) because verifiable reward is gated on format_ok. This is intentional — model must learn format before getting answer credit.
- **DS stage 2 risk**: 7B full-rank past step 31 without hanging (previous code runs hung ~step 50). Monitoring closely.
- **7B full learning faster**: training_correct_rate 0.83 vs LoRA's 0.41 — more samples have positive scores (mostly format reward, not sycophancy yet).
