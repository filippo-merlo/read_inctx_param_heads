# Atlas heads through Head-JLens

This project re-runs the head identification of *The Atlas of In-Context Learning*
(Kahardipraja et al., NeurIPS 2025; [code](https://github.com/pkhdipraja/in-context-atlas))
on `meta-llama/Llama-3.1-8B-Instruct`. It finds the top-100 in-context and parametric heads
and the top-40 task and retrieval heads. It then reads those heads with the J-head and
Logit-head readouts of [Head-JLens](https://github.com/filippo-merlo/head_level_jlens).

- `results/heads.json` lists each identified head: layer, head, category, rank and the
  official scores. `results/head_map.png` shows them on a layer × head map.
- `explore_heads.py` is a marimo notebook. It sets out the method, compares the re-run with
  the paper, and lets you read any identified head at any position of an NQ-Swap or
  free-text prompt.
- `heads_from_readings.py` is a marimo notebook. It asks whether the readings alone tell
  in-context from parametric heads, reading all 1,024 heads at the position and answer token
  that AttnLRP explains.
- `lexical_task_heads.py` is a marimo notebook. It applies the lexical-task-head criterion of
  [Yang et al. (2026)](https://arxiv.org/abs/2604.22027) to the same readings, with GPT-written
  task words per question (`data/task_terms/`), and compares the result with the Atlas task heads.

## Setup

The official code and Head-JLens need different `transformers` versions, so there are two
environments:

```bash
scripts/setup_atlas.sh   # official repo at a pinned commit + its uv env, JDK 11, spaCy model, NQ-Swap
uv sync                  # notebook env: head_level_jlens (../head_level_jlens) + marimo
```

## Run

```bash
scripts/run_all.sh full 0 2 3 5                     # official pipeline on four GPUs -> results/heads.json
.venv/bin/python scripts/decode_categories.py --gpu 0 # per-head top-k tokens for explore_heads.py
.venv/bin/python scripts/decode_targets.py --gpu 0    # all heads at the AttnLRP targets, for heads_from_readings.py
HEADS_GPU=0 .venv/bin/marimo edit explore_heads.py
.venv/bin/marimo edit heads_from_readings.py           # no GPU needed
.venv/bin/python scripts/annotate_task_terms.py       # task words per question; needs AZURE_OPENAI_API_KEY and _ENDPOINT
.venv/bin/marimo edit lexical_task_heads.py            # no GPU needed
```

`scripts/run_atlas.sh` runs single stages; `scripts/run_all.sh smoke` runs every stage on
16 examples. Official outputs land in `data/full/results/`, logs in `logs/full/`.
