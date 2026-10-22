"""Every head's reading at the official AttnLRP target, under J-head and Logit-head.

The official relevance step (``preprocessing/get_relevance.py``) explains, for each correctly
answered item, the logit of the first located answer token, at the position that predicts
it: the prompt followed by the generated tokens before that answer token. This script reads
all 1,024 heads at that same position, for the same items of the counterfactual and the
de-duplicated closed-book runs. It stores each head's top-k tokens and the rank and score
of the target token. ``heads_from_readings.py`` analyses the output.

    .venv/bin/python scripts/decode_targets.py --gpu 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data" / "full" / "results"
OUT_DIR = ROOT / "data" / "decode"
RUNS = {"cf": "cf", "cb": "cb_dedupe"}  # output key -> official run
N_LAYERS = N_HEADS = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k", type=int, default=10, help="tokens kept per head")
    parser.add_argument("--gpu", type=str, default="0", help="nvidia-smi index")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # before torch is imported
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from tqdm import tqdm

    from heads_reading import LENSES, Reader

    heads = [(layer, head) for layer in range(N_LAYERS) for head in range(N_HEADS)]
    reader = Reader.load()
    arrays = {}
    for key, run in RUNS.items():
        generation = torch.load(OFFICIAL / "ablation" / run / "generation.pt")
        bem = torch.load(OFFICIAL / "preprocessed" / "llama" / run / "bem.pt", weights_only=False)
        # The official keep_correct_prediction (preprocessing/score_bem.py): a BEM hit with
        # an answer token located in the generation.
        items = [i for i in range(len(bem["hit"])) if bem["hit"][i] and len(bem["overlap"][i]) > 0]
        n = len(items)
        top = np.zeros((n, len(heads), len(LENSES), args.k), dtype=np.int32)
        rank = np.zeros((n, len(heads), len(LENSES)), dtype=np.int32)
        logit = np.zeros((n, len(heads), len(LENSES)), dtype=np.float32)
        targets = np.zeros(n, dtype=np.int32)
        positions = np.zeros(n, dtype=np.int32)
        for row, i in enumerate(tqdm(items, desc=key)):
            offset, target = bem["overlap"][i][0]
            input_ids = list(generation["input_ids"][i]) + list(generation["generated_ids"][i][:offset])
            result = reader.read(input_ids, len(input_ids) - 1, heads, k=args.k, target=target)
            for j, lens in enumerate(LENSES):
                top[row, :, j] = result[lens][0].numpy()
                rank[row, :, j] = result["target"][lens][0].numpy()
                logit[row, :, j] = result["target"][lens][1].numpy()
            targets[row], positions[row] = target, len(input_ids) - 1
        arrays |= {
            f"{key}_items": np.array(items), f"{key}_target": targets, f"{key}_position": positions,
            f"{key}_top": top, f"{key}_rank": rank, f"{key}_logit": logit,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"targets_all_heads_top{args.k}.npz"
    np.savez_compressed(out, heads=np.array(heads), lenses=np.array(LENSES), **arrays)
    print("->", out)


if __name__ == "__main__":
    main()
