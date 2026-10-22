"""Read the Atlas-of-ICL heads through Head-JLens (J-head) and the Logit-head baseline.

Shared by ``scripts/decode_categories.py`` and ``explore_heads.py``. The model wrapper, the
fitted lens, the activation hook and both readouts are head_level_jlens's, used unchanged;
this module only feeds them the official NQ-Swap inputs and the identified heads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
HEADS_JSON = ROOT / "results" / "heads.json"
RUNS_DIR = ROOT / "data" / "full" / "results" / "ablation"

LENS_KEY = "llama3.1-8b-it"
CONDITIONS = {"closed-book": "cb", "oracle": "oracle", "counterfactual": "cf"}
CATEGORIES = ("in_context", "parametric", "task", "retrieval")
LENSES = ("jhead", "logithead")


def load_heads(path: Path = HEADS_JSON) -> dict[str, list[dict]]:
    """``{category: [entry, ...]}`` in rank order, as written by ``export_heads.py``."""
    by_category: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for entry in json.loads(path.read_text())["heads"]:
        by_category[entry["category"]].append(entry)
    for entries in by_category.values():
        entries.sort(key=lambda e: e["rank"])
    return by_category


def load_run(condition: str) -> list[dict]:
    """One condition's official inference records with their per-item metrics, in NQ-Swap order."""
    path = RUNS_DIR / CONDITIONS[condition] / "llama_nq-swap_inference_metrics.json"
    records = json.loads(path.read_text())
    return [records[str(i)] for i in range(len(records))]


@dataclass
class Reader:
    """Llama-3.1-8B-Instruct with its fitted ``J^z`` and ``W_O``, ready to read heads."""

    model: object
    spec: object
    jz: dict[int, torch.Tensor]
    w_o: dict[int, torch.Tensor]

    @classmethod
    def load(cls) -> Reader:
        from src.config import LENSES as LENS_SPECS
        from src.config import jz_path
        from src.heads import o_proj_weight
        from src.model import load_jz, load_model

        spec = LENS_SPECS[LENS_KEY]
        # Eager attention so that each attention module returns its weights.
        model = load_model(spec, attn_implementation="eager")
        lens = load_jz(spec, jz_path(spec, n_prompts=spec.jz_n_prompts))
        jz = {l: lens.jacobians[l].to(model.input_device) for l in spec.jz_source_layers}
        w_o = {l: o_proj_weight(model, l) for l in spec.jz_source_layers}
        return cls(model, spec, jz, w_o)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def encode(self, text: str) -> list[int]:
        """Token ids with BOS, as the official runs tokenize their prompts."""
        return self.tokenizer(text).input_ids

    @torch.no_grad()
    def read(
        self,
        input_ids: list[int],
        position: int,
        heads: list[tuple[int, int]],
        *,
        k: int = 10,
        attention_layer: int | None = None,
        target: int | None = None,
    ) -> dict:
        """Top-k tokens of each ``(layer, head)`` at ``position`` under both lenses.

        Returns ``{"jhead": (ids, logits), "logithead": (ids, logits)}`` with ``[n_heads, k]``
        tensors in the order of ``heads``, plus ``"attention"``: the ``[n_heads, seq, seq]``
        weights of ``attention_layer`` when one is given, and ``"target"``:
        ``{lens: (rank, logit)}`` with ``[n_heads]`` tensors when a ``target`` token id is
        given. The rank is 0-based: the number of tokens the head scores strictly higher.
        """
        from src.hooks import AnchorRecorder
        from src.readout import all_heads_transported

        layers = sorted({layer for layer, _ in heads})
        ids = torch.tensor([input_ids], device=self.model.input_device)
        captured: dict[str, torch.Tensor] = {}
        hook = None
        if attention_layer is not None:
            attention = self.model.layers[attention_layer].self_attn

            def keep_weights(_module, _inputs, outputs):
                captured["attention"] = outputs[1][0].float().cpu()

            hook = attention.register_forward_hook(keep_weights)
        try:
            with AnchorRecorder(self.model.layers, zin_at=layers) as recorder:
                self.model.forward(ids)
        finally:
            if hook is not None:
                hook.remove()

        result: dict = {}
        for lens in LENSES:
            per_layer = {
                layer: all_heads_transported(
                    self.jz[layer],
                    recorder.activations[("zin", layer)][0, position],
                    self.spec,
                    arm=lens,
                    w_o=self.w_o[layer],
                )
                for layer in layers
            }
            vectors = torch.stack([per_layer[layer][head] for layer, head in heads])
            logits = self.model.unembed(vectors).float()
            values, token_ids = logits.topk(k, dim=-1)
            result[lens] = (token_ids.cpu(), values.cpu())
            if target is not None:
                target_logit = logits[:, target]
                rank = (logits > target_logit[:, None]).sum(-1)
                result.setdefault("target", {})[lens] = (rank.cpu(), target_logit.cpu())
        if "attention" in captured:
            result["attention"] = captured["attention"]
        return result
