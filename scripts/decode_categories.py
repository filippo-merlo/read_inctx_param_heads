"""Top-k tokens of every identified head, under J-head and Logit-head, on NQ-Swap examples.

The same seeded sample of examples is read under closed-book, oracle and counterfactual
prompts, taken verbatim (token ids) from the official inference runs. Each head is read at
the last prompt token, the position whose prediction is the first answer token. The
notebook aggregates the stored token ids per head category.

    .venv/bin/python scripts/decode_categories.py --gpu 0
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "decode"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=500, help="NQ-Swap examples per condition")
    parser.add_argument("--k", type=int, default=10, help="tokens kept per head")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0", help="nvidia-smi index")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # before torch is imported
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from tqdm import tqdm

    from heads_reading import CONDITIONS, LENSES, Reader, load_heads, load_run

    by_category = load_heads()
    heads = sorted({(e["layer"], e["head"]) for entries in by_category.values() for e in entries})
    runs = {condition: load_run(condition) for condition in CONDITIONS}
    n_items = {len(records) for records in runs.values()}
    if len(n_items) != 1:
        raise ValueError(f"conditions differ in length: {n_items}")
    examples = sorted(random.Random(args.seed).sample(range(n_items.pop()), args.n))

    reader = Reader.load()
    arrays = {}
    for condition, records in runs.items():
        token_ids = np.zeros((args.n, len(heads), len(LENSES), args.k), dtype=np.int32)
        for row, item in enumerate(tqdm(examples, desc=condition)):
            input_ids = records[item]["input_ids"]
            result = reader.read(input_ids, len(input_ids) - 1, heads, k=args.k)
            for j, lens in enumerate(LENSES):
                token_ids[row, :, j] = result[lens][0].numpy()
        arrays[CONDITIONS[condition]] = token_ids

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"top{args.k}_n{args.n}_seed{args.seed}.npz"
    np.savez_compressed(
        out,
        examples=np.array(examples),
        heads=np.array(heads),
        lenses=np.array(LENSES),
        **arrays,
    )
    print("->", out)


if __name__ == "__main__":
    main()
