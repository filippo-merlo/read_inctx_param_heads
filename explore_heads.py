import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Atlas heads through Head-JLens")


@app.cell(hide_code=True)
def _():
    import os

    # Pin the GPU before torch is imported. This node mixes A100s and an H100, so CUDA's
    # default device order would not match nvidia-smi indices.
    GPU = os.environ.get("HEADS_GPU", "0")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    return (GPU,)


@app.cell(hide_code=True)
def _(GPU):
    import html
    import json
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    import heads_reading as hr

    _ = GPU
    ROOT = Path(hr.__file__).resolve().parent
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    def token_text(token_id: int) -> str:
        """A token's text; tokens that are incomplete UTF-8 bytes keep their byte-level
        spelling, so that different ones are not all shown as the same replacement mark."""
        text = tokenizer.decode([token_id])
        return tokenizer.convert_ids_to_tokens(token_id) if "\ufffd" in text else text

    return Counter, ROOT, alt, html, hr, json, mo, np, pd, token_text, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # In-context and parametric attention heads in Llama-3.1-8B-Instruct, read through Head-JLens

    ## 1 Introduction

    When a language model answers a question with a passage in its prompt, the answer can
    come from the passage or from what the model learned in training. Kahardipraja et al.
    (2025) locate attention heads that serve each source. *In-context heads* contribute most
    when the model answers from the prompt, and *parametric heads* when it answers from
    memory. Among in-context heads they further separate *task heads*, which attend to the
    question, from *retrieval heads*, which attend to the answer in the passage.

    This notebook does two things. First, it re-runs their head identification on
    Llama-3.1-8B-Instruct with the official code and compares the outcome with the paper
    (Section 3). Second, it reads what each identified head writes, using two per-head
    readouts from Merlo (2026): the per-head logit lens (Logit-head) and Head-JLens
    (J-head) (Section 4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 Methods

    ### 2.1 Model and data

    The model is Llama-3.1-8B-Instruct, with 32 layers of 32 attention heads, 1,024 heads
    in total. It runs in bfloat16 with greedy decoding and at most 20 new tokens, without
    a chat template, as in the original study.

    The data is NQ-Swap (Longpre et al., 2021): 4,746 Natural Questions items, each with
    its original Wikipedia passage and a *counterfactual* copy of that passage in which
    the answer entity has been replaced by another one. Questions are formatted with the
    template of Ram et al. (2023), with the first letter lower-cased and a question mark
    added if missing:

    | Condition | Prompt | Reference answer |
    |---|---|---|
    | closed-book | `Answer these questions:\nQ: {question}\nA:` | original answer |
    | oracle | `{original passage}\n\nBased on this text, answer these questions:\nQ: {question}\nA:` | original answer |
    | counterfactual | `{counterfactual passage}\n\nBased on this text, answer these questions:\nQ: {question}\nA:` | substituted answer |

    ### 2.2 Answer quality: recall

    The model's response is its generation up to the first newline. Recall (Adlakha et
    al., 2024) is the fraction of the reference answer's tokens that appear in the
    response, after lower-casing and removing punctuation and articles. It is maximised
    over the reference answers of an item and averaged over items. Recall is used instead
    of exact match because instruction-tuned models tend to answer verbosely.

    ### 2.3 In-context and parametric heads

    For each head $h$, let $\mathbf z^h_i \in \mathbb R^{d_h}$ be its output at
    position $i$ before the output projection. AttnLRP (Achtibat et al., 2024) propagates
    the logit of a target token $y_t$ back through the network and assigns a relevance to
    every activation. The contribution of a head is its positive relevance summed over
    positions and dimensions:

    $$\mathcal R^h(y_t) = \sum_{i=1}^{S}\sum_{k=1}^{d_h} \max\!\big(\mathcal R(\mathbf z^h_i \mid y_t)_k,\,0\big).$$

    The target is the first token of the answer. Only items whose response matches the
    reference answer are used. A response matches when every entity of the reference has
    a counterpart in the response with sentence-embedding cosine similarity above 0.9,
    and that counterpart can be located among the generated tokens. The two settings
    contrasted are counterfactual prompts, explained at the substituted answer
    $y_{\mathrm{cf}}$, and closed-book prompts on de-duplicated questions, explained at
    the original answer $y_{\mathrm{gold}}$:

    $$\mathcal D^h = \mathbb E_{\text{counterfactual}}\big[\mathcal R^h(y_{\mathrm{cf}})\big] - \mathbb E_{\text{closed-book}}\big[\mathcal R^h(y_{\mathrm{gold}})\big].$$

    The 100 heads with the highest $\mathcal D^h$ are the in-context heads. The 100 with
    the lowest are the parametric heads.

    ### 2.4 Task and retrieval heads

    On counterfactual prompts, the relevance of each attention weight $A^h_{i,j}$ is
    summed over query positions $i$. This gives, for every key token $j$, how much the
    head's reading of that token contributes to the answer:

    $$\rho^h_j = \sum_{i=1}^{S} \max\!\big(\mathcal R(A^h_{i,j}\mid y_t),\,0\big),\qquad
    \rho^h_{\text{task}} = \sum_{j\in J_{\text{question}}} \rho^h_j,\qquad
    \rho^h_{\text{ret}} = \sum_{j\in J_{\text{answer}}} \rho^h_j,$$

    where $J_{\text{question}}$ are the question's tokens and $J_{\text{answer}}$ the
    tokens of the substituted answer inside the passage. Both scores are averaged over
    the correctly answered items. The official code skips prompts longer than 1,500
    tokens and prompts in which it cannot locate the answer sentence. The 40 heads with
    the highest averages are the task heads and the retrieval heads.

    Selection follows the official code. Heads are ranked by score, then early layers are
    removed: layers 0–1 for task heads, and flat index $32\,\ell + h < 168$ (layers 0–4
    and heads 0–7 of layer 5) for retrieval heads. The first 40 that remain are kept.

    ### 2.5 Reading a head: Logit-head and J-head

    Let $\mathbf c_{\ell,h} = W_{O,\ell}^{(h)}\mathbf z_{\ell,h}$ be what head $h$ of
    layer $\ell$ adds to the residual stream at a position. Both readouts turn a vector
    into vocabulary scores with the model's own output head: the final RMSNorm, then the
    unembedding.

    - **Logit-head** decodes $\mathbf c_{\ell,h}$ directly. This is the per-head logit
      lens; the original study uses the same quantity to score retrieval heads.
    - **J-head** (Head-JLens; Merlo, 2026) first transports the head's output to the
      final layer: $\mathbf r_{\ell,h} = J^z_{\ell}[:,\mathcal C_h]\,\mathbf z_{\ell,h}$.
      Here $J^z_\ell$ is the Jacobian of the final residual stream with respect to the
      concatenated head outputs of layer $\ell$, averaged over 1,000 WikiText-103
      prompts as in the Jacobian lens of Gurnee et al. (2026), and $\mathcal C_h$ are
      head $h$'s columns. J-head therefore reads a head through the computation of the
      layers that follow it.

    Section 4 reads heads at a chosen token position. Both readouts use the fitted lens
    and code of Merlo (2026) unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(hr, json):
    heads_data = json.loads(hr.HEADS_JSON.read_text())
    heads_by_category = hr.load_heads()
    runs = {condition: hr.load_run(condition) for condition in hr.CONDITIONS}
    return heads_by_category, heads_data, runs


@app.cell(hide_code=True)
def _(heads_data, mo, np, pd, runs):
    # Table 4 of Kahardipraja et al. (2025), Llama-3.1-8B-Instruct on NQ-Swap, recall (%).
    PAPER_RECALL = {"oracle": 87.67, "counterfactual": 68.73, "DPR": 46.57, "closed-book": 51.64}

    _rows = []
    for _condition in ("oracle", "counterfactual", "DPR", "closed-book"):
        _rerun = _condition in runs
        _rows.append({
            "Condition": _condition,
            "Items": f"{len(runs[_condition]):,}" if _rerun else "",
            "Recall, this run (%)": (
                f"{100 * np.mean([r['recall'] for r in runs[_condition]]):.2f}"
                if _rerun else "not re-run"
            ),
            "Recall, paper (%)": f"{PAPER_RECALL[_condition]:.2f}",
        })
    table4 = pd.DataFrame(_rows)
    _n = heads_data["examples"]
    mo.vstack([
        mo.md(r"""
        ## 3 Reproduced results

        ### 3.1 Answer quality

        Table 1 compares answer recall with Table 4 of the original study. The DPR
        condition, which retrieves a Wikipedia passage instead of using the given one,
        was not re-run: it plays no part in head identification.
        """),
        mo.md("**Table 1.** Recall of Llama-3.1-8B-Instruct on NQ-Swap."),
        mo.ui.table(table4, selection=None, pagination=False, show_column_summaries=False),
        mo.md(f"""
        Head identification used the items answered correctly:
        {_n["counterfactual"]["answered_correctly"]:,} of
        {_n["counterfactual"]["prompts"]:,} counterfactual prompts and
        {_n["closed_book_deduplicated"]["answered_correctly"]:,} of
        {_n["closed_book_deduplicated"]["prompts"]:,} de-duplicated closed-book questions.
        Task and retrieval relevance was computed on
        {_n["counterfactual"]["task_retrieval_scored"]:,} of the correct counterfactual items.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(alt, heads_data, pd):
    _scores = heads_data["per_head_scores"]
    _n_heads = len(_scores["in_context_D"])
    _members: dict[int, set] = {i: set() for i in range(_n_heads)}
    _rank: dict[tuple, int] = {}
    for _e in heads_data["heads"]:
        _flat = 32 * _e["layer"] + _e["head"]
        _members[_flat].add(_e["category"])
        _rank[(_flat, _e["category"])] = _e["rank"]

    map_frame = pd.DataFrame({
        "layer": [i // 32 for i in range(_n_heads)],
        "head": [i % 32 for i in range(_n_heads)],
        "type": [
            "in-context" if "in_context" in _members[i]
            else "parametric" if "parametric" in _members[i] else "neither"
            for i in range(_n_heads)
        ],
        "task head": ["task" in _members[i] for i in range(_n_heads)],
        "retrieval head": ["retrieval" in _members[i] for i in range(_n_heads)],
        "D": _scores["in_context_D"],
        "task relevance": _scores["task_relevance"],
        "retrieval relevance": _scores["retrieval_relevance"],
    })

    _tooltip = [
        "layer", "head", "type", "task head", "retrieval head",
        alt.Tooltip("D", format=".4g"),
        alt.Tooltip("task relevance", format=".4g"),
        alt.Tooltip("retrieval relevance", format=".4g"),
    ]
    # Layout of Figure 1 in Kahardipraja et al. (2025): layer 0 at the top, one dot per
    # head; in-context and parametric heads small, task and retrieval heads large.
    # Blue, orange and aqua pass the all-pairs colour-blindness check; shape and size
    # carry the role as a second cue.
    _all = list(range(32))  # every layer and head, so empty rows keep their place
    _x = alt.X("head:O", title="Head", scale=alt.Scale(domain=_all),
               axis=alt.Axis(labelAngle=0, values=list(range(0, 32, 5))))
    _y = alt.Y("layer:O", title="Layer", scale=alt.Scale(domain=_all),
               axis=alt.Axis(values=list(range(0, 32, 5))))
    _types = alt.Chart(map_frame).transform_filter("datum.type != 'neither'").mark_point(
        filled=True, size=30
    ).encode(
        x=_x, y=_y,
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["in-context", "parametric"], range=["#2a78d6", "#eb6834"]),
            legend=alt.Legend(title="Head type", orient="top"),
        ),
        opacity=alt.Opacity(
            "type:N", scale=alt.Scale(domain=["in-context", "parametric"], range=[0.45, 0.9]),
            legend=None,
        ),
        tooltip=_tooltip,
    )
    _roles = alt.Chart(map_frame).transform_fold(
        ["task head", "retrieval head"], as_=["role", "is_role"]
    ).transform_filter("datum.is_role").mark_point(
        filled=True, size=130, opacity=1, stroke="white", strokeWidth=1.5
    ).encode(
        x=_x, y=_y,
        color=alt.Color(
            "role:N",
            scale=alt.Scale(domain=["task head", "retrieval head"], range=["#2a78d6", "#1baf7a"]),
            legend=alt.Legend(title="Top-40", orient="top"),
        ),
        shape=alt.Shape(
            "role:N",
            scale=alt.Scale(domain=["task head", "retrieval head"], range=["circle", "diamond"]),
            legend=None,
        ),
        tooltip=_tooltip,
    )

    _layer_sums = map_frame.groupby("layer", as_index=False)[
        ["task relevance", "retrieval relevance"]
    ].sum()

    def _bars(column, colour):
        return alt.Chart(_layer_sums).mark_bar(color=colour, cornerRadiusEnd=4).encode(
            x=alt.X(f"{column}:Q", title=f"Summed {column}"),
            y=alt.Y("layer:O", scale=alt.Scale(domain=list(range(32))), axis=None),
            tooltip=["layer", alt.Tooltip(f"{column}:Q", format=".4g")],
        ).properties(width=130, height=480)

    functional_map = alt.hconcat(
        alt.layer(_types, _roles).resolve_scale(color="independent").properties(width=480, height=480),
        _bars("task relevance", "#2a78d6"),
        _bars("retrieval relevance", "#1baf7a"),
    ).configure_view(stroke=None)
    return functional_map, map_frame


@app.cell(hide_code=True)
def _(ROOT, functional_map, mo):
    functional_map.save(str(ROOT / "results" / "head_map.png"), scale_factor=2)
    mo.vstack([
        mo.md(r"""
        ### 3.2 Functional map

        Figure 1 shows the identified heads in the layout of Figure 1 of the original
        study. On the right are the layer-wise sums, over all 32 heads of each layer,
        of the mean task relevance and the mean retrieval relevance.
        """),
        functional_map,
        mo.md(
            "**Figure 1.** In-context (small, light blue) and parametric (small orange) heads, "
            "with the top-40 task heads (large blue circles) and retrieval heads (large "
            "aqua diamonds). Hover a head for its scores; the tables below list them."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(heads_by_category, map_frame, mo, pd):
    _quarters = [(0, 7), (8, 15), (16, 23), (24, 31)]
    _counts = pd.DataFrame([
        {"Category": _c.replace("_", "-")} | {
            f"Layers {_a}–{_b}": sum(_a <= _e["layer"] <= _b for _e in _entries)
            for _a, _b in _quarters
        }
        for _c, _entries in heads_by_category.items()
    ])
    _sums = map_frame.groupby("layer")[["task relevance", "retrieval relevance"]].sum()
    _first_quarter_parametric = sum(_e["layer"] <= 7 for _e in heads_by_category["parametric"])
    mo.vstack([
        mo.md("**Table 2.** Identified heads per quarter of the network."),
        mo.ui.table(_counts, selection=None, pagination=False, show_column_summaries=False),
        mo.md(f"""
        *Comparison with the original study.* Kahardipraja et al. (2025) describe the
        functional map of Llama-3.1-8B-Instruct as a concentrated band of in-context heads
        in the middle layers, flanked by parametric heads in the early and late layers
        (their Appendix C.1). Task relevance rises over the early layers and peaks in the
        middle layers, while retrieval relevance peaks in deeper layers (their Section 6.1).

        The re-run reproduces the middle band: most in-context heads, and most task heads,
        lie in layers 8–15 (Table 2). Summed task relevance peaks at layer
        {int(_sums["task relevance"].idxmax())} and summed retrieval relevance deeper, at
        layer {int(_sums["retrieval relevance"].idxmax())}; most retrieval heads lie in
        the last quarter. Parametric heads match only in part. Almost all of them lie in
        the second half of the network, but the early flank is nearly absent: only
        {_first_quarter_parametric} of the 100 lie in the first quarter, whereas the
        original map also places parametric heads in the first layers.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(heads_by_category, mo, pd):
    _frames = {
        _c.replace("_", "-"): pd.DataFrame([
            {
                "rank": _e["rank"], "layer": _e["layer"], "head": _e["head"],
                "score": _e["score"], "D": _e["scores"]["in_context_D"],
                "task relevance": _e["scores"]["task_relevance"],
                "retrieval relevance": _e["scores"]["retrieval_relevance"],
            }
            for _e in _entries
        ])
        for _c, _entries in heads_by_category.items()
    }
    mo.vstack([
        mo.md(
            "The identified heads, from `results/heads.json`. The score is the one each "
            "category is ranked by: $\\mathcal D$, $-\\mathcal D$, $\\rho_{\\text{task}}$ "
            "or $\\rho_{\\text{ret}}$."
        ),
        mo.ui.tabs({
            _name: mo.ui.table(_f, selection=None, page_size=10, show_column_summaries=False)
            for _name, _f in _frames.items()
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Reading the heads

    ### 4.1 One head at one position

    Choose a prompt, a head and a token position. The two tables list the head's
    top-scoring tokens under J-head and Logit-head. The prompt below them is shaded by
    the head's attention from the chosen position.
    """)
    return


@app.cell(hide_code=True)
def _(hr):
    reader = hr.Reader.load()
    return (reader,)


@app.cell(hide_code=True)
def _(hr, mo):
    source = mo.ui.radio(["NQ-Swap example", "free text"], value="NQ-Swap example", label="Prompt")
    condition = mo.ui.dropdown(list(hr.CONDITIONS), value="counterfactual", label="Condition")
    item = mo.ui.number(start=0, stop=4745, step=1, value=0, label="Example (0–4745)")
    free_text = mo.ui.text_area(
        value="Answer these questions:\nQ: what is the capital of france?\nA:",
        label="Free text", rows=4, full_width=True,
    )
    category = mo.ui.dropdown(
        {c.replace("_", "-"): c for c in hr.CATEGORIES}, value="retrieval", label="Head category"
    )
    top_k = mo.ui.slider(5, 30, value=10, step=1, label="Top-k", show_value=True)
    return category, condition, free_text, item, source, top_k


@app.cell(hide_code=True)
def _(category, heads_by_category, mo):
    _options = {
        f"L{e['layer']} H{e['head']}  (rank {e['rank']}, score {e['score']:.4g})":
            (e["layer"], e["head"])
        for e in heads_by_category[category.value]
    }
    head = mo.ui.dropdown(_options, value=next(iter(_options)), label="Head")
    head
    return (head,)


@app.cell(hide_code=True)
def _(condition, free_text, item, reader, runs, source):
    if source.value == "free text":
        record = None
        input_ids = reader.encode(free_text.value)
    else:
        record = runs[condition.value][item.value]
        input_ids = record["input_ids"]
    return input_ids, record


@app.cell(hide_code=True)
def _(input_ids, mo, token_text):
    _options = {f"{i}: {token_text(t)!r}": i for i, t in enumerate(input_ids)}
    position = mo.ui.dropdown(_options, value=list(_options)[-1], label="Position")
    return (position,)


@app.cell(hide_code=True)
def _(category, condition, free_text, head, item, mo, position, record, source, top_k):
    _prompt_controls = [condition, item] if source.value == "NQ-Swap example" else [free_text]
    _info = mo.md("")
    if record is not None:
        _info = mo.md(
            f"**Question:** {record['question']}  \n"
            f"**Reference answer(s):** {record['gold_ans']}  \n"
            f"**Model answer:** {record['cleaned_ans'].strip()!r} (recall {record['recall']:.2f})"
        )
    mo.vstack([
        mo.hstack([source, *_prompt_controls], justify="start", align="end"),
        _info,
        mo.hstack([category, head, position, top_k], justify="start", align="end"),
    ])
    return


@app.cell(hide_code=True)
def _(head, html, input_ids, mo, pd, position, reader, token_text, tokenizer, top_k):
    _layer, _head = head.value
    _pos = position.value
    _result = reader.read(input_ids, _pos, [head.value], k=top_k.value, attention_layer=_layer)

    def _table(lens):
        _ids, _values = _result[lens]
        return pd.DataFrame({
            "rank": range(1, top_k.value + 1),
            "token": [repr(token_text(int(t))) for t in _ids[0]],
            "score": [round(float(v), 3) for v in _values[0]],
        })

    _weights = _result["attention"][_head, _pos, : _pos + 1].numpy()
    _scale = _weights[1:].max() if _pos > 0 else _weights.max()
    _spans = []
    for _i, _t in enumerate(input_ids[: _pos + 1]):
        _text = html.escape(tokenizer.decode([_t])).replace("\n", "⏎<br>")
        _alpha = min(1.0, float(_weights[_i] / _scale)) if _scale > 0 else 0.0
        _outline = "outline: 2px solid #2a78d6;" if _i == _pos else ""
        _spans.append(
            f'<span title="{_i}: {_weights[_i]:.4f}" style="background: rgba(42,120,214,{_alpha:.3f});'
            f'{_outline} border-radius: 3px;">{_text}</span>'
        )
    _attention_html = mo.Html(
        '<div style="font-family: var(--monospace-font, monospace); font-size: 13px; '
        'line-height: 1.7; white-space: pre-wrap; max-height: 360px; overflow-y: auto;">'
        + "".join(_spans) + "</div>"
    )
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("**J-head**"), mo.ui.table(_table("jhead"), selection=None, pagination=False, show_column_summaries=False)]),
            mo.vstack([mo.md("**Logit-head**"), mo.ui.table(_table("logithead"), selection=None, pagination=False, show_column_summaries=False)]),
        ], widths="equal"),
        mo.md(
            f"**Attention of L{_layer} H{_head} from position {_pos}.** Shading is "
            f"proportional to the attention weight, scaled to the largest weight after "
            f"the first token; the first token (`<|begin_of_text|>`) receives "
            f"{_weights[0]:.3f}. Hover a token for its weight."
        ),
        _attention_html,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Most common decoded tokens per head category

    For 500 NQ-Swap examples, sampled once with a fixed seed and the same in every
    condition, each identified head is read at the last prompt token, whose prediction
    is the first answer token. A token's share is the fraction of (example, head) pairs
    of a category in which the token is among the head's top-$k$. Tokens are compared
    after stripping surrounding whitespace and lower-casing; tokens that are incomplete
    UTF-8 byte sequences are shown in the tokenizer's byte-level spelling.
    """)
    return


@app.cell(hide_code=True)
def _(ROOT, np):
    decoded = np.load(ROOT / "data" / "decode" / "top10_n500_seed0.npz")
    return (decoded,)


@app.cell(hide_code=True)
def _(mo):
    agg_k = mo.ui.slider(1, 10, value=5, step=1, label="Top-k per head", show_value=True)
    agg_n = mo.ui.slider(5, 30, value=15, step=1, label="Tokens shown", show_value=True)
    mo.hstack([agg_k, agg_n], justify="start")
    return agg_k, agg_n


@app.cell(hide_code=True)
def _(Counter, agg_k, decoded, heads_by_category, hr, pd, token_text):
    _head_index = {(int(l), int(h)): i for i, (l, h) in enumerate(decoded["heads"])}
    _vocab_cache: dict[int, str] = {}

    def _word(token_id):
        if token_id not in _vocab_cache:
            _raw = token_text(token_id)
            _vocab_cache[token_id] = _raw.strip().lower() or repr(_raw)
        return _vocab_cache[token_id]

    _rows = []
    for _cond_name, _key in hr.CONDITIONS.items():
        _ids = decoded[_key][..., : agg_k.value]  # [examples, heads, lens, k]
        for _category, _entries in heads_by_category.items():
            _cols = [_head_index[(e["layer"], e["head"])] for e in _entries]
            _pairs = _ids.shape[0] * len(_cols)
            for _j, _lens in enumerate(hr.LENSES):
                _counts = Counter()
                for _per_head in _ids[:, _cols, _j, :].reshape(_pairs, -1):
                    _counts.update({_word(int(t)) for t in _per_head})
                for _w, _c in _counts.items():
                    _rows.append({
                        "condition": _cond_name, "category": _category.replace("_", "-"),
                        "lens": {"jhead": "J-head", "logithead": "Logit-head"}[_lens],
                        "token": _w, "share": _c / _pairs,
                    })
    token_shares = pd.DataFrame(_rows)
    return (token_shares,)


@app.cell(hide_code=True)
def _(agg_n, alt, hr, mo, token_shares):
    def _panel(frame, colour):
        _top = frame.nlargest(agg_n.value, "share")
        return alt.Chart(_top).mark_bar(color=colour, cornerRadiusEnd=4).encode(
            x=alt.X("share:Q", title="Share of (example, head) pairs", axis=alt.Axis(format="%")),
            y=alt.Y("token:N", sort="-x", title=None),
            tooltip=["token", alt.Tooltip("share:Q", format=".1%")],
        ).properties(width=220, height=agg_n.value * 16)

    _lens_colour = {"J-head": "#2a78d6", "Logit-head": "#eb6834"}
    _tabs = {}
    for _cond in hr.CONDITIONS:
        _f = token_shares[token_shares["condition"] == _cond]
        _rows = []
        for _category in ("in-context", "parametric", "task", "retrieval"):
            _rows.append(alt.hconcat(*[
                _panel(_f[(_f["category"] == _category) & (_f["lens"] == _lens)], _colour)
                .properties(title=f"{_category} · {_lens}")
                for _lens, _colour in _lens_colour.items()
            ]))
        _tabs[_cond] = alt.vconcat(*_rows)
    mo.ui.tabs(_tabs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Achtibat, R., et al. (2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. *ICML*.
    - Adlakha, V., et al. (2024). Evaluating correctness and faithfulness of instruction-following models for question answering. *TACL*.
    - Gurnee, W., Sofroniew, N., Pearce, A., et al. (2026). [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html). Transformer Circuits Thread.
    - Kahardipraja, P., Achtibat, R., Wiegand, T., Samek, W., & Lapuschkin, S. (2025). [The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](https://arxiv.org/abs/2505.15807). *NeurIPS*.
    - Longpre, S., et al. (2021). Entity-based knowledge conflicts in question answering. *EMNLP*.
    - Merlo, F. (2026). Reading Attention Heads Through the Jacobian Lens.
    - Ram, O., et al. (2023). In-context retrieval-augmented language models. *TACL*.
    """)
    return


if __name__ == "__main__":
    app.run()
