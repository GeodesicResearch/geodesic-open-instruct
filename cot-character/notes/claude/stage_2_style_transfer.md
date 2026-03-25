# Stage 2: Style Transfer & Data Assembly

## High-Level Goal

Take the corpus of correct reasoning traces from Stage 1 and, for each of the 9 constitutions, produce a version of those traces where the inner monologue reflects that character's emotional and motivational patterns. The logical steps and final answers must be preserved — only the "texture" of the reasoning changes. Then format everything into SFT-ready training data.

This is the most creatively demanding stage of the pipeline. The style transfer must be substantial enough that the model can learn the pattern from examples alone (the constitution text is never shown during training), but realistic enough that the traces don't become cartoonish or break logical coherence.

## Context

The 9 constitutions are stored in `cot-character/constitutions/*.yaml` (see `cot-character/CLAUDE.md` for format details). Each defines a character's inner monologue through a list of prose trait descriptions — covering what the character feels, why they help, how they experience difficulty, etc.

The key design constraint: during SFT (Stage 3), the model never sees the constitution text. It must internalize the reasoning style purely from the styled traces. This means the style transfer must embed the character naturally into the reasoning flow, not as explicit self-description or meta-commentary.

## Interface

### Inputs
- `data/source_traces/` from Stage 1 — JSONL with prompt, reasoning_trace, final_answer, domain, source_dataset
- `constitutions/*.yaml` — the 9 character definitions
- `data_generation/prompts/` — prompt template(s) for the style-transfer task (to be designed at this stage)

### Outputs
- `data/sft_datasets/{character_name}.jsonl` — one file per character, in HuggingFace messages format:
  ```json
  {
    "messages": [
      {"role": "user", "content": "<the original prompt>"},
      {"role": "assistant", "content": "<think>\n<styled reasoning trace>\n</think>\n<final answer>"}
    ]
  }
  ```
- No constitution text anywhere in the output files
- Roughly same corpus size per character

### Key Design Artifact
- The style-transfer prompt template in `data_generation/prompts/` — this is what tells the teacher model how to rewrite traces. It must:
  - Present the constitution traits
  - Present the source trace
  - Instruct the model to rewrite the inner monologue in that style
  - Explicitly instruct preservation of logical steps and final answer
  - Explicitly instruct against meta-commentary, cartoonishness, and constitution text leakage

## The Style Transfer Task

The teacher model (a non-thinking instruct model — see `cot-character/CLAUDE.md` for rationale) receives a prompt roughly structured as:

1. Here is a character description (the constitution traits)
2. Here is a reasoning trace for a problem
3. Rewrite the reasoning trace so the inner monologue reflects this character
4. Keep the logical steps and final answer identical
5. Do not mention the character description or comment on it — just *be* the character while reasoning

The choice to use a non-thinking model is deliberate: we want this framed as a style transfer / rewriting task, not as a "generate CoT" task. Thinking models tend to be bad at controlling their own CoT.

## Checks

All checks should be applied to the final `data/sft_datasets/{character_name}.jsonl` output:

- **Answer preservation**: the final answer in the styled trace matches the source trace's final answer (exact match or semantic equivalence depending on domain)
- **Style transfer quality** — manual inspection of ~20 traces per character:
  - The inner monologue genuinely reflects the constitution's patterns
  - No meta-commentary on the constitution (e.g. "As an anxious thinker, I now feel worried about...")
  - Not cartoonish or over-exaggerated — retains realism and subtlety
  - The reasoning actually reaches the stated final answer (no disconnected conclusions)
- **Within-character variation**: traces for a single character show meaningful differences in expression across different problems. The character should feel like a person with consistent traits but natural variation, not a template being stamped repeatedly
- **No constitution leakage**: grep the output files for constitution text fragments — none should appear
- **Format validity**: every record parses as valid messages format with user + assistant roles
- **Tokenized length**: all examples fit within the model's context window
- **Coherent reasoning**: logical steps aren't broken by the rewriting (the proof still proves, the debugging still debugs)
- **Corpus balance**: roughly similar dataset sizes across all 9 characters

## Key Decisions to Make at This Stage

- **Prompt template design**: the exact wording of the style-transfer prompt is critical and will likely need iteration. Start with a reasonable first draft, generate a small batch, inspect, and refine
- **Teacher model choice**: which non-thinking instruct model to use. Should be capable enough for nuanced rewriting but doesn't need to be a reasoning specialist. Whatever is already available in the infrastructure is preferred
- **Batch size**: how many traces to style-transfer per character. Depends on SFT data requirements and generation cost
- **Filtering strategy**: how aggressively to filter. Some traces may not style-transfer well (e.g. very short traces may not have enough room for character expression). May want to filter these out
