"""Export the heads identified by the official Atlas-of-ICL pipeline.

Runs in the official env (``scripts/run_atlas.sh <mode> export``). Reads the outputs of
``rank_mean_retrieval_histogram.py`` and ``filter_heads_composition.py``. The top-40 task
and retrieval heads come from the official ``get_head_ids`` in
``analysis_functional/{task_heads/task_fv.py, retrieval_heads/retrieval_needle.py}``,
called as those scripts call it by default (``ctx=False``), so nothing is re-implemented.

Writes ``heads.json`` (path given as the only argument): one entry per (head, category), with the head's layer and
index, its rank in the category, and the official scores of that head.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch
from configs import GlobalPath

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "third_party" / "in-context-atlas"

MODEL = "llama"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
COMMIT = "a30e468140"
N_HEADS_PER_LAYER = 32
TOP = {"in_context": 100, "parametric": 100, "task": 40, "retrieval": 40}


def official_get_head_ids(subdir: str, module: str) -> list[int]:
    """The official ``get_head_ids(model, ctx=False)`` of one functional-analysis script."""
    sys.path.insert(0, str(ATLAS / "analysis_functional" / subdir))
    try:
        return importlib.import_module(module).get_head_ids(MODEL, False)
    finally:
        sys.path.pop(0)


def main(out_path: Path) -> None:
    pre = GlobalPath().result_path / "preprocessed" / MODEL
    abl = GlobalPath().result_path / "ablation"

    in_context = torch.load(pre / "in_context_heads/histogram/mean_histogram_mode=sum_pos_rel.pt")
    parametric = torch.load(pre / "parametric_heads/histogram/mean_histogram_mode=sum_pos_rel.pt")
    filtered = torch.load(pre / "cf" / "head_filtered.pt")

    n_heads = in_context["scores"].numel()
    d = torch.empty(n_heads)
    d[in_context["indices"]] = in_context["scores"]  # D = E_cf[R^h] - E_cb[R^h]
    rho_task = filtered["question"].mean(0)
    rho_ret = filtered["target"].mean(0)

    ids = {
        "in_context": in_context["indices"][: TOP["in_context"]].tolist(),
        "parametric": parametric["indices"][: TOP["parametric"]].tolist(),
        "task": official_get_head_ids("task_heads", "task_fv")[: TOP["task"]],
        "retrieval": official_get_head_ids("retrieval_heads", "retrieval_needle")[: TOP["retrieval"]],
    }
    ranking_score = {"in_context": d, "parametric": -d, "task": rho_task, "retrieval": rho_ret}

    heads = []
    for category, flat_ids in ids.items():
        for rank, flat in enumerate(flat_ids, start=1):
            heads.append({
                "layer": flat // N_HEADS_PER_LAYER,
                "head": flat % N_HEADS_PER_LAYER,
                "category": category,
                "rank": rank,
                "score": ranking_score[category][flat].item(),
                "scores": {
                    "in_context_D": d[flat].item(),
                    "task_relevance": rho_task[flat].item(),
                    "retrieval_relevance": rho_ret[flat].item(),
                },
            })

    def n_items(run: str) -> int:
        return len(json.loads((abl / run / "llama_nq-swap_inference.json").read_text()))

    def n_relevance(run: str) -> int:
        return torch.load(pre / run / "attn_pre_out.pt")["sum_pos_rel"].shape[0]

    out = {
        "model": MODEL_ID,
        "code": f"https://github.com/pkhdipraja/in-context-atlas@{COMMIT}",
        "examples": {
            "counterfactual": {"prompts": n_items("cf"), "answered_correctly": n_relevance("cf"),
                               "task_retrieval_scored": filtered["question"].shape[0]},
            "closed_book_deduplicated": {"prompts": n_items("cb_dedupe"),
                                         "answered_correctly": n_relevance("cb_dedupe")},
        },
        "per_head_scores": {
            "in_context_D": d.tolist(),
            "task_relevance": rho_task.tolist(),
            "retrieval_relevance": rho_ret.tolist(),
        },
        "heads": heads,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print({c: len(v) for c, v in ids.items()}, "->", out_path)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
