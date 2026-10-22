import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Heads from their readings")


@app.cell(hide_code=True)
def _():
    import json
    from collections import Counter

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    import heads_reading as hr

    ROOT = hr.ROOT
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    def token_text(token_id: int) -> str:
        """A token's text; tokens that are incomplete UTF-8 bytes keep their byte-level
        spelling, so that different ones are not all shown as the same replacement mark."""
        text = tokenizer.decode([token_id])
        return tokenizer.convert_ids_to_tokens(token_id) if "�" in text else text

    return Counter, ROOT, alt, hr, json, mo, np, pd, token_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Telling in-context from parametric heads by their readings

    ## 1 Question

    Kahardipraja et al. (2025) identify in-context and parametric heads with AttnLRP:
    a head's relevance to the answer on counterfactual prompts, minus its relevance on
    closed-book prompts (`explore_heads.py`, Section 2.3). This notebook asks whether
    the heads' readings are enough to do the same. A reading is the list of tokens that
    a head's output promotes under J-head or Logit-head. The question has two parts:
    do the readings tell the 100 in-context heads from the 100 parametric heads, and do
    they find those heads among all 1,024?
    """)
    return


@app.cell(hide_code=True)
def _(mo, n_items):
    mo.md(rf"""
    ## 2 Methods

    ### 2.1 Items and reading position

    The items are the ones the official relevance step explains:
    {n_items["cf"]:,} correctly answered counterfactual prompts and
    {n_items["cb"]:,} correctly answered de-duplicated closed-book questions. For each
    item, the target $y_t$ is the first answer token located in the model's generation,
    and the reading position is the one whose next-token prediction is $y_t$: the prompt
    followed by the generated tokens before $y_t$. This is where AttnLRP explains the
    logit of $y_t$. All 1,024 heads are read there with J-head and Logit-head, as defined
    in `explore_heads.py`, Section 2.5.

    ### 2.2 Readings

    Each measure uses a head's top-$k$ tokens at the reading position.

    - **Decoded tokens per category.** A token's share is the fraction of
      (item, head) pairs of a category in which the token is among the head's top-$k$.
      Tokens are compared after stripping surrounding whitespace and lower-casing.
    - **Answer rate.** $a^h_{{c}}$ is the fraction of items of condition $c$ in which
      $y_t$ is among head $h$'s top-$k$. The answer contrast
      $\Delta a^h = a^h_{{\text{{counterfactual}}}} - a^h_{{\text{{closed-book}}}}$
      is the reading-based counterpart of $\mathcal D^h$: it takes the same two
      conditions and the same target, but asks only whether the head promotes the target,
      not how much it contributes to its logit.
    - **Passage share.** On counterfactual items, the fraction of a head's top-$k$
      tokens that occur in the counterfactual prompt but not in the closed-book prompt of
      the same question, that is, tokens that only the passage brings. Unlike the answer
      rate, it needs no answer.

    ### 2.3 Comparison with the Atlas heads

    Each per-head measure $m$ is compared with the heads of `results/heads.json` in three
    ways:

    - **AUC**: the probability that a random in-context head has a higher $m$ than a
      random parametric head, ties counting one half. 0.5 is chance; 1 is perfect
      separation.
    - **Spearman correlation** between $m$ and $\mathcal D$ over all 1,024 heads.
    - **Recovery**: how many of the 100 heads with the highest $m$ are in-context heads,
      and how many of the 100 with the lowest $m$ are parametric heads. Heads tied at the
      cut are counted in proportion, so that the count does not depend on how ties are
      ordered. 100 heads drawn at random hold on average about 10 heads of each set.

    In-context heads lie mostly in the middle layers and parametric heads in the late
    ones (`explore_heads.py`, Table 2). The table therefore also scores layer depth itself,
    with earlier heads ranked as more in-context. It gets an AUC and a correlation but no
    recovery count, since the in-context heads are not the earliest heads.
    """)
    return


@app.cell(hide_code=True)
def _(ROOT, hr, json, np):
    heads_data = json.loads(hr.HEADS_JSON.read_text())
    _npz = np.load(ROOT / "data" / "decode" / "targets_all_heads_top10.npz")
    decoded = {key: _npz[key] for key in _npz.files}  # decompress once
    n_items = {c: len(decoded[f"{c}_items"]) for c in ("cf", "cb")}

    D = np.array(heads_data["per_head_scores"]["in_context_D"])
    layer = decoded["heads"][:, 0]
    members = [set() for _ in range(len(D))]
    for _e in heads_data["heads"]:
        members[32 * _e["layer"] + _e["head"]].add(_e["category"])
    in_context = np.array(["in_context" in m for m in members])
    parametric = np.array(["parametric" in m for m in members])
    return D, decoded, in_context, layer, members, n_items, parametric


@app.cell(hide_code=True)
def _(decoded, hr, np):
    # For every counterfactual item, which of each head's top-10 tokens only the passage brings.
    _cf_run, _cb_run = hr.load_run("counterfactual"), hr.load_run("closed-book")
    in_passage = np.stack([
        np.isin(_top, list(set(_cf_run[i]["input_ids"]) - set(_cb_run[i]["input_ids"])))
        for _top, i in zip(decoded["cf_top"], decoded["cf_items"])
    ])  # [items, heads, lenses, 10]
    return (in_passage,)


@app.cell(hide_code=True)
def _(mo):
    top_k = mo.ui.slider(1, 10, value=10, step=1, label="Top-k per head", show_value=True)
    return (top_k,)


@app.cell(hide_code=True)
def _(D, decoded, hr, in_context, in_passage, layer, members, np, parametric, pd, top_k):
    _k = top_k.value
    answer_rate = {c: (decoded[f"{c}_rank"] < _k).mean(0) for c in ("cf", "cb")}  # [heads, lenses]
    passage_share = in_passage[..., :_k].mean(-1).mean(0)  # [heads, lenses]
    _lens_name = {"jhead": "J-head", "logithead": "Logit-head"}

    def _auc(a, b):
        a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
        return float((a > b).mean() + 0.5 * (a == b).mean())

    def _recovered(m, members_mask, n=100):
        """Members among the n heads with the highest m, ties at the cut counted in proportion."""
        cut = np.sort(m)[::-1][n - 1]
        above, tied = m > cut, m == cut
        return members_mask[above].sum() + (n - above.sum()) * members_mask[tied].mean()

    def _row(name, m):
        return {
            "Measure": name,
            "AUC": round(_auc(m[in_context], m[parametric]), 3),
            "Spearman with D": round(float(pd.Series(m).rank().corr(pd.Series(D).rank())), 3),
            "In-context among top 100": round(float(_recovered(m, in_context)), 1),
            "Parametric among bottom 100": round(float(_recovered(-m, parametric)), 1),
        }

    _rows = []
    for _j, _lens in enumerate(hr.LENSES):
        _rows.append(_row(f"Answer contrast Δa, {_lens_name[_lens]}",
                          answer_rate["cf"][:, _j] - answer_rate["cb"][:, _j]))
    for _j, _lens in enumerate(hr.LENSES):
        _rows.append(_row(f"Passage share, {_lens_name[_lens]}", passage_share[:, _j]))
    _layer_row = _row("Layer (earlier = in-context)", -layer.astype(float))
    _layer_row["In-context among top 100"] = _layer_row["Parametric among bottom 100"] = None
    _rows.append(_layer_row)
    evaluation = pd.DataFrame(_rows)

    _role = lambda m: ", ".join(r for r in ("task", "retrieval") if r in m)
    per_head = pd.DataFrame({
        "layer": layer,
        "head": decoded["heads"][:, 1],
        "Atlas": ["in-context" if "in_context" in m else "parametric" if "parametric" in m
                  else "neither" for m in members],
        "role": [_role(m) for m in members],
        "D": D,
        **{f"answer rate cf, {_lens_name[l]}": answer_rate["cf"][:, j] for j, l in enumerate(hr.LENSES)},
        **{f"answer rate cb, {_lens_name[l]}": answer_rate["cb"][:, j] for j, l in enumerate(hr.LENSES)},
        **{f"passage share, {_lens_name[l]}": passage_share[:, j] for j, l in enumerate(hr.LENSES)},
    })
    return evaluation, per_head


@app.cell(hide_code=True)
def _(mo, top_k):
    mo.vstack([
        mo.md(r"""
        ## 3 Results

        Every result below uses the top-$k$ tokens of each head; the slider sets $k$.
        The text describes $k = 10$.
        """),
        top_k,
    ])
    return


@app.cell(hide_code=True)
def _(Counter, decoded, hr, in_context, np, parametric, pd, token_text, top_k):
    _vocab_cache: dict[int, str] = {}

    def _word(token_id):
        if token_id not in _vocab_cache:
            _raw = token_text(token_id)
            _vocab_cache[token_id] = _raw.strip().lower() or repr(_raw)
        return _vocab_cache[token_id]

    _rows = []
    for _cond, _key in (("counterfactual", "cf"), ("closed-book", "cb")):
        _top = decoded[f"{_key}_top"][..., : top_k.value]  # [items, heads, lenses, k]
        for _category, _mask in (("in-context", in_context), ("parametric", parametric)):
            _cols = np.flatnonzero(_mask)
            _pairs = _top.shape[0] * len(_cols)
            for _j, _lens in enumerate(hr.LENSES):
                _counts = Counter()
                for _per_head in _top[:, _cols, _j, :].reshape(_pairs, -1):
                    _counts.update({_word(int(t)) for t in _per_head})
                for _w, _c in _counts.most_common(30):
                    _rows.append({
                        "condition": _cond, "category": _category,
                        "lens": {"jhead": "J-head", "logithead": "Logit-head"}[_lens],
                        "token": _w, "share": _c / _pairs,
                    })
    token_shares = pd.DataFrame(_rows)
    return (token_shares,)


@app.cell(hide_code=True)
def _(alt, mo, token_shares):
    _n_shown = 15
    _colour = {"in-context": "#2a78d6", "parametric": "#eb6834"}

    def _panel(frame, colour):
        _top = frame.nlargest(_n_shown, "share")
        return alt.Chart(_top).mark_bar(color=colour, cornerRadiusEnd=4).encode(
            x=alt.X("share:Q", title="Share of (item, head) pairs", axis=alt.Axis(format="%")),
            y=alt.Y("token:N", sort="-x", title=None),
            tooltip=["token", alt.Tooltip("share:Q", format=".1%")],
        ).properties(width=220, height=_n_shown * 16)

    _tabs = {}
    for _cond in ("counterfactual", "closed-book"):
        _f = token_shares[token_shares["condition"] == _cond]
        _tabs[_cond] = alt.vconcat(*[
            alt.hconcat(*[
                _panel(_f[(_f["category"] == _category) & (_f["lens"] == _lens)], _colour[_category])
                .properties(title=f"{_category} · {_lens}")
                for _lens in ("J-head", "Logit-head")
            ])
            for _category in ("in-context", "parametric")
        ])
    mo.vstack([
        mo.md(r"""
        ### 3.1 Decoded tokens per category

        In both conditions and under both readouts, the most common tokens of in-context
        and parametric heads are punctuation and formatting tokens (Figure 1). No answer
        content appears: the answer changes from item to item, so it never becomes a common
        token. The two categories differ in which tokens they share. Read through J-head,
        in-context heads share ellipses, colons and question marks across many items, while
        parametric heads share commas, full stops, parentheses, function words and a few
        topic words such as "film" and "movie". Read through Logit-head, in-context heads give mostly word fragments.
        This follows their depth rather than their role: Logit-head leaves out the layers
        after a head, and most in-context heads lie in the middle of the network.
        """),
        mo.ui.tabs(_tabs),
        mo.md(
            "**Figure 1.** The 15 most common decoded tokens of in-context heads (blue) and "
            "parametric heads (orange), per readout and condition."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(alt, mo, pd, per_head):
    _colour = alt.Scale(domain=["in-context", "parametric", "neither"],
                        range=["#2a78d6", "#eb6834", "#a3a3a0"])

    def _scatter(lens):
        _x, _y = f"answer rate cf, {lens}", f"answer rate cb, {lens}"
        _tooltip = ["layer", "head", "Atlas", "role", alt.Tooltip("D:Q", format=".4g"),
                    alt.Tooltip(f"{_x}:Q", format=".1%"), alt.Tooltip(f"{_y}:Q", format=".1%")]
        _top = max(per_head[_x].max(), per_head[_y].max())
        _diagonal = alt.Chart(pd.DataFrame({"v": [0, _top]})).mark_line(color="#a3a3a0", strokeDash=[4, 4], strokeWidth=1).encode(x="v:Q", y="v:Q")
        _base = alt.Chart(per_head).encode(
            x=alt.X(f"{_x}:Q", title="Answer rate, counterfactual", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, _top])),
            y=alt.Y(f"{_y}:Q", title="Answer rate, closed-book", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, _top])),
            color=alt.Color("Atlas:N", scale=_colour, legend=alt.Legend(title="Atlas head", orient="top", symbolOpacity=1)),
            tooltip=_tooltip,
        )
        _neither = _base.transform_filter("datum.Atlas == 'neither'").mark_circle(size=18, opacity=0.5)
        _atlas = _base.transform_filter("datum.Atlas != 'neither'").mark_circle(
            size=60, opacity=0.9, stroke="white", strokeWidth=1
        )
        return alt.layer(_diagonal, _neither, _atlas).properties(title=lens, width=300, height=300)

    mo.vstack([
        mo.md(r"""
        ### 3.2 The answer in the readings

        Most heads never place the target among their top-$k$ in either condition, and
        this holds for most in-context and most parametric heads too: they form the cluster
        at the origin of Figure 2. The few heads that promote the answer often fall on the
        expected side of the diagonal. Under J-head, the heads that most often promote the
        counterfactual answer are in-context heads that are also retrieval heads: L16H1,
        L17H24, L18H20, L27H20 and L18H23. Under Logit-head only the late one, L27H20,
        stands out; the middle-layer ones promote the answer far less often. On closed-book
        items, the heads that most often promote the answer are parametric under both
        readouts: L29H11, L31H7, L29H9 and L28H5.

        Some heads promote the answer in both conditions. L29H11, the first-ranked
        parametric head, does so more often on counterfactual items than on closed-book
        ones, so the answer contrast places it on the in-context side. L29H8, which is in
        neither Atlas set, also promotes the answer often in both conditions. A reading
        shows that a head writes the answer. Whether that answer came from the passage or
        from memory shows only in the comparison between conditions, and there the heads
        that write the answer in both conditions cancel out.
        """),
        alt.hconcat(_scatter("J-head"), _scatter("Logit-head")).configure_view(stroke=None),
        mo.md(
            "**Figure 2.** Each dot is one of the 1,024 heads: how often the target is among "
            "its top-$k$ on counterfactual items (x) and on closed-book items (y). Heads below "
            "the dashed diagonal promote the answer more often when it comes from the passage. "
            "Hover a head for its scores."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(alt, mo, per_head):
    _colour = alt.Scale(domain=["in-context", "parametric", "neither"],
                        range=["#2a78d6", "#eb6834", "#a3a3a0"])

    def _strip(lens):
        _y = f"passage share, {lens}"
        _base = alt.Chart(per_head).encode(
            x=alt.X("layer:Q", title="Layer", scale=alt.Scale(domain=[0, 31]),
                    axis=alt.Axis(values=list(range(0, 32, 5)))),
            y=alt.Y(f"{_y}:Q", title="Passage share of top-k", axis=alt.Axis(format="%")),
            color=alt.Color("Atlas:N", scale=_colour, legend=alt.Legend(title="Atlas head", orient="top", symbolOpacity=1)),
            tooltip=["layer", "head", "Atlas", "role", alt.Tooltip("D:Q", format=".4g"),
                     alt.Tooltip(f"{_y}:Q", format=".1%")],
        )
        _neither = _base.transform_filter("datum.Atlas == 'neither'").mark_circle(size=18, opacity=0.5)
        _atlas = _base.transform_filter("datum.Atlas != 'neither'").mark_circle(
            size=60, opacity=0.9, stroke="white", strokeWidth=1
        )
        return alt.layer(_neither, _atlas).properties(title=lens, width=300, height=260)

    mo.vstack([
        mo.md(r"""
        ### 3.3 Passage tokens in the readings

        The passage share runs against the Atlas split. Under both readouts, the heads
        whose readings hold the most passage tokens are late parametric heads: L30H7, L30H5
        and L30H4 (Figure 3). A parametric head in the Atlas sense contributes more to
        closed-book answers than to counterfactual ones; this does not stop it from
        promoting passage tokens when there is a passage. Under Logit-head, passage tokens
        are nearly absent from the readings of the first half of the network and become
        common in the last layers, so the measure mostly orders heads by depth, in the
        opposite direction from the Atlas split.
        """),
        alt.hconcat(_strip("J-head"), _strip("Logit-head")).configure_view(stroke=None),
        mo.md(
            "**Figure 3.** Each dot is one of the 1,024 heads: the share of its top-$k$ "
            "tokens on counterfactual items that only the passage brings, by layer."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(evaluation, mo, per_head):
    mo.vstack([
        mo.md(r"""
        ### 3.4 Agreement with the Atlas heads

        Neither measure reproduces the Atlas sets (Table 1). The answer contrast
        separates in-context from parametric heads better than chance, more so under J-head
        than under Logit-head, but well below what layer depth alone achieves, and it
        hardly correlates with $\mathcal D$ across all heads. Among all 1,024 heads, the
        100 with the highest answer contrast hold only somewhat more in-context heads than a
        random draw would; the 100 with the lowest recover parametric heads better, about a
        third of them. The passage share is
        at chance under J-head and runs against the Atlas split under Logit-head.
        """),
        mo.md("**Table 1.** Each measure against the Atlas heads (Section 2.3)."),
        mo.ui.table(evaluation, selection=None, pagination=False, show_column_summaries=False),
        mo.md("**Table 2.** All measures for every head; sort by any column."),
        mo.ui.table(per_head.round(4), selection=None, page_size=10, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Conclusion

    The readings single out a handful of heads of each kind: through J-head, the
    middle-layer retrieval heads that write the counterfactual answer, and through both
    readouts, the late parametric heads that write the closed-book answer. They do not
    recover the Atlas sets. Most identified heads never place the answer among their top
    tokens, and the heads that do often write it whatever its source. The Atlas criterion
    weighs how much each head contributes to the answer logit in each condition, and a
    head's top tokens do not carry that.

    ## References

    - Achtibat, R., et al. (2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. *ICML*.
    - Kahardipraja, P., Achtibat, R., Wiegand, T., Samek, W., & Lapuschkin, S. (2025). [The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](https://arxiv.org/abs/2505.15807). *NeurIPS*.
    - Merlo, F. (2026). Reading Attention Heads Through the Jacobian Lens.
    """)
    return


if __name__ == "__main__":
    app.run()
