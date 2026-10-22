import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Lexical task heads")


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
    TERMS = ROOT / "data" / "task_terms" / "gpt-5.5.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    def token_text(token_id: int) -> str:
        """A token's text; tokens that are incomplete UTF-8 bytes keep their byte-level
        spelling, so that different ones are not all shown as the same replacement mark."""
        text = tokenizer.decode([token_id])
        return tokenizer.convert_ids_to_tokens(token_id) if "�" in text else text

    return Counter, ROOT, TERMS, alt, hr, json, mo, np, pd, token_text, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lexical task heads and the Atlas heads in Llama-3.1-8B-Instruct

    ## 1 Question

    Yang et al. (2026) find *lexical task heads*: attention heads whose output, read with
    the logit lens at the position that predicts the answer, promotes words that describe
    the task rather than its answer. For "The capital city of Japan is", such a head writes
    "cities"; for an antonym prompt, "opposite". Kahardipraja et al. (2025) find *task
    heads* in another way: with AttnLRP, as the heads whose attention to the question
    contributes most to the answer (`explore_heads.py`, Section 2.4). This notebook applies
    the first criterion to the NQ-Swap items of the second and asks whether it finds the
    same heads, and how the heads it finds relate to the other Atlas sets: the in-context,
    parametric and retrieval heads. It applies the criterion under both per-head readouts,
    Logit-head and J-head.
    """)
    return


@app.cell(hide_code=True)
def _(annotations, mo, n_items, terms_per_question):
    _models = ", ".join(sorted({a["model"] for a in annotations.values()}))
    mo.md(rf"""
    ## 2 Methods

    ### 2.1 Items and reading position

    The items are the {n_items:,} correctly answered counterfactual prompts that the
    official relevance step explains, the same items from which the Atlas task heads are
    identified. For each item, the target $y_t$ is the first answer token located in the
    model's generation. The reading position is the one whose next-token prediction is
    $y_t$: the prompt followed by the generated tokens before $y_t$. This is the
    counterpart of the last prompt token in Yang et al., whose prompts end right before the
    answer, and it is where AttnLRP explains the logit of $y_t$. All 1,024 heads are read
    there with J-head and Logit-head, as defined in `explore_heads.py`, Section 2.5. The
    Logit-head readout is the decoding of Yang et al.: the head's output, the final RMSNorm,
    then the unembedding.

    ### 2.2 Task words

    Yang et al. match a head's top tokens against a list of words that describe each of
    their tasks. NQ-Swap has no fixed tasks, so each of the {len(annotations):,} distinct
    questions gets its own list. A language model ({_models}) names the task the question
    asks for and lists single words that describe it or the kind of answer it asks for,
    without the answer and without names of specific entities
    (`scripts/annotate_task_terms.py`). Yang et al. check their lists in the same way with
    LLM-written ones, GPT-5.4 Mini among them (their Appendix O). Their prompt is not
    published, so the prompt is ours; it shows five of their hand-written lists as
    examples. A question gets {terms_per_question:.1f} words on average.

    Each word is turned into token ids as in their code: the word as written, with a
    leading space, in upper case, capitalised, in lower case, and with a leading ".", "_"
    or "-". Every variant that is a single token is kept. The target $y_t$ is removed from
    an item's set, so that a head that writes the answer is not counted as writing the
    task.

    ### 2.3 Lexical task heads

    The criterion is that of Yang et al., with their parameters $n = 1$, $k = 10$ and
    $p = 10\%$. Head $h$ matches item $i$ when at least one of its top-10 tokens is in the
    item's set. Its score $s^h$ is the fraction of items it matches, and it is a lexical
    task head when $s^h \geq 0.10$. Yang et al. apply the threshold per task, over its
    first 100 correctly answered prompts. Here every question is its own task, so the
    threshold applies over all items. The criterion is applied to each readout separately.

    ### 2.4 Comparison with the Atlas heads

    `results/heads.json` holds four Atlas sets: the 100 in-context and the 100 parametric
    heads (`explore_heads.py`, Section 2.3), and the 40 task and the 40 retrieval heads
    (Section 2.4). Task and retrieval heads can also be in-context heads, so a head can
    belong to more than one set. Each lexical set is compared with each Atlas set in three
    ways:

    - **Overlap**: how many lexical task heads are in the Atlas set.
    - **Overlap expected from layers**: each Atlas set has its own layer profile, so some
      overlap follows from depth alone. A set with the same number of heads in each layer
      as the lexical set, drawn at random within layers, shares on average
      $\sum_\ell n_\ell a_\ell / 32$ heads with the Atlas set, where $n_\ell$ and $a_\ell$
      are the two sets' head counts in layer $\ell$.
    - **Recovery**: how many heads of the Atlas set are among the $m$ heads with the
      highest $s^h$, where $m$ is the Atlas set's size. Heads tied at the cut are counted
      in proportion. $m$ heads drawn at random hold on average $m^2 / 1{{,}}024$ of them:
      about 9.8 for the in-context and parametric sets, about 1.6 for the task and
      retrieval sets.
    """)
    return


@app.cell(hide_code=True)
def _(ROOT, TERMS, hr, json, np):
    heads_data = json.loads(hr.HEADS_JSON.read_text())
    _npz = np.load(ROOT / "data" / "decode" / "targets_all_heads_top10.npz")
    decoded = {key: _npz[key] for key in ("heads", "cf_items", "cf_target", "cf_top")}
    n_items = len(decoded["cf_items"])
    annotations = {a["question"]: a for a in map(json.loads, TERMS.read_text().splitlines())}
    _records = hr.load_run("counterfactual")
    item_questions = [_records[i]["question"] for i in decoded["cf_items"]]

    layer = decoded["heads"][:, 0]
    members = [set() for _ in range(len(layer))]
    task_rank = {}
    for _e in heads_data["heads"]:
        _flat = 32 * _e["layer"] + _e["head"]
        members[_flat].add(_e["category"])
        if _e["category"] == "task":
            task_rank[_flat] = _e["rank"]
    atlas_sets = {
        name: np.array([category in m for m in members])
        for name, category in (("task", "task"), ("retrieval", "retrieval"),
                               ("in-context", "in_context"), ("parametric", "parametric"))
    }
    atlas_task = atlas_sets["task"]
    atlas_role = np.array([
        "task" if "task" in m else "retrieval" if "retrieval" in m
        else "other in-context" if "in_context" in m else "parametric" if "parametric" in m
        else "none" for m in members
    ])
    return (annotations, atlas_role, atlas_sets, atlas_task, decoded, item_questions, layer,
            n_items, task_rank)


@app.cell(hide_code=True)
def _(annotations, decoded, item_questions, np, tokenizer):
    def expand(words):
        """Token ids of each word's single-token variants, as in Yang et al.'s
        ``convert_expand_candidate_strs_to_token_ids_nnsight``."""
        ids = set()
        for _w in words:
            for _variant in (_w, " " + _w, _w.upper(), _w.capitalize(), _w.lower(),
                             "." + _w, "_" + _w, "-" + _w):
                _t = tokenizer.encode(_variant, add_special_tokens=False)
                if len(_t) == 1:
                    ids.add(_t[0])
        return ids

    _by_question = {q: expand(a["terms"]) for q, a in annotations.items()}
    terms_per_question = float(np.mean([len(a["terms"]) for a in annotations.values()]))
    _item_ids = [
        list(_by_question[q] - {int(t)}) for q, t in zip(item_questions, decoded["cf_target"])
    ]
    matched = np.stack([
        np.isin(_top, _ids) for _top, _ids in zip(decoded["cf_top"], _item_ids)
    ])  # [items, heads, lenses, 10]
    score = matched.any(-1).mean(0)  # [heads, lenses]
    lexical = score >= 0.10
    return lexical, matched, score, terms_per_question


@app.cell(hide_code=True)
def _(atlas_role, atlas_sets, hr, layer, lexical, np, pd, score):
    lens_name = {"jhead": "J-head", "logithead": "Logit-head"}

    def _recovered(m, members_mask, n):
        """Members among the n heads with the highest m, ties at the cut counted in proportion."""
        cut = np.sort(m)[::-1][n - 1]
        above, tied = m > cut, m == cut
        return members_mask[above].sum() + (n - above.sum()) * members_mask[tied].mean()

    summary = pd.DataFrame([
        {
            "Readout": lens_name[_lens],
            "Lexical task heads": int(lexical[:, _j].sum()),
            "In no Atlas set": int((atlas_role[lexical[:, _j]] == "none").sum()),
            "Shared with the other readout": int((lexical[:, _j] & lexical[:, 1 - _j]).sum()),
        }
        for _j, _lens in enumerate(hr.LENSES)
    ])

    _rows = []
    for _name, _mask in atlas_sets.items():
        _m = int(_mask.sum())
        for _j, _lens in enumerate(hr.LENSES):
            _set = lexical[:, _j]
            _rows.append({
                "Atlas set": _name,
                "Atlas heads": _m,
                "Readout": lens_name[_lens],
                "Overlap": int((_set & _mask).sum()),
                "Overlap expected from layers": round(float(
                    np.bincount(layer[_set], minlength=32) @ np.bincount(layer[_mask], minlength=32) / 32), 1),
                "Among the m highest s": round(float(_recovered(score[:, _j], _mask, _m)), 1),
                "Among m random heads": round(_m * _m / len(_mask), 1),
            })
    comparison = pd.DataFrame(_rows)
    return comparison, lens_name, summary


@app.cell(hide_code=True)
def _(atlas_role, decoded, hr, lens_name, lexical, pd, score, task_rank):
    per_head = pd.DataFrame({
        "layer": decoded["heads"][:, 0],
        "head": decoded["heads"][:, 1],
        "Atlas": atlas_role,
        "Atlas task rank": [task_rank.get(h) for h in range(len(atlas_role))],
        **{f"s, {lens_name[l]}": score[:, j] for j, l in enumerate(hr.LENSES)},
        **{f"lexical, {lens_name[l]}": lexical[:, j] for j, l in enumerate(hr.LENSES)},
    })
    return (per_head,)


@app.cell(hide_code=True)
def _(comparison, mo, summary):
    mo.vstack([
        mo.md(r"""
        ## 3 Results

        ### 3.1 Agreement with the Atlas heads

        The lexical-task criterion picks mostly different heads from all four Atlas sets.
        Under both readouts, most lexical task heads are in no Atlas set (Table 1). The two
        readouts largely agree with each other: almost every Logit-head lexical task head is
        also a J-head one, and J-head adds heads in layers 13 to 17, where Logit-head
        readings give mostly word fragments (`heads_from_readings.py`, Section 3.1).

        Two Atlas sets show some agreement (Table 2). Under J-head, the overlap with the
        task heads is about twice what the two sets' layer profiles alone would give, and
        the 40 heads with the highest score hold several task heads where a random draw
        would hold one or two. Under Logit-head, the task overlap is two heads, L16H25 and
        L15H21, little more than depth alone predicts. The parametric heads are the other
        set: under both readouts, the 100 heads with the highest score hold about twice as
        many parametric heads as a random draw. Under J-head the parametric overlap also
        exceeds what depth predicts; under Logit-head depth accounts for it. Parametric
        heads lie late in the network, where many lexical task heads also lie, so part of
        this agreement follows from depth.

        The in-context heads are, if anything, rarer among the high-scoring heads than in a
        random draw, and the retrieval heads overlap no more than depth predicts. For the
        retrieval heads this is partly by construction: they write the answer, and the
        answer token is removed from every item's task words.
        """),
        mo.md("**Table 1.** Each readout's lexical task heads."),
        mo.ui.table(summary, selection=None, pagination=False, show_column_summaries=False),
        mo.md("**Table 2.** Each readout's lexical task heads against each Atlas set "
              "(Section 2.4). $m$ is the size of the Atlas set."),
        mo.ui.table(comparison, selection=None, pagination=False, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(alt, hr, lens_name, mo, per_head):
    _sets = alt.Scale(domain=["Atlas task head", "Lexical task head"], range=["#2a78d6", "#eb6834"])

    def _map(lens):
        _base = alt.Chart(per_head).encode(
            x=alt.X("layer:Q", title="Layer", scale=alt.Scale(domain=[-0.5, 31.5]),
                    axis=alt.Axis(values=list(range(0, 32, 5)), grid=False)),
            y=alt.Y("head:Q", title="Head", scale=alt.Scale(domain=[-0.5, 31.5]),
                    axis=alt.Axis(values=list(range(0, 32, 5)), grid=False)),
            tooltip=["layer", "head", "Atlas", "Atlas task rank",
                     alt.Tooltip(f"s, {lens}:Q", format=".1%")],
        )
        _all = _base.mark_circle(size=6, color="#d4d4d0", opacity=1)
        _atlas = _base.transform_filter("datum.Atlas == 'task'").transform_calculate(
            set="'Atlas task head'"
        ).mark_circle(size=60, opacity=1).encode(
            color=alt.Color("set:N", scale=_sets, legend=alt.Legend(title=None, orient="top")))
        _lexical = _base.transform_filter(f"datum['lexical, {lens}']").transform_calculate(
            set="'Lexical task head'"
        ).mark_point(shape="circle", filled=False, size=150, strokeWidth=2, opacity=1).encode(
            color=alt.Color("set:N", scale=_sets))
        return alt.layer(_all, _atlas, _lexical).properties(title=lens, width=300, height=300)

    mo.vstack([
        mo.md(r"""
        ### 3.2 Where the lexical and the Atlas task heads lie

        The Atlas task heads lie mostly in layers 12 to 16, with a few in the first layers.
        The lexical task
        heads lie a few layers later, mostly between layers 13 and 23, with a handful in the
        last layers (Figure 1). The heads in both sets sit where the two ranges meet, in
        layers 13 to 16.
        """),
        alt.hconcat(*[_map(lens_name[l]) for l in hr.LENSES]).configure_view(stroke=None),
        mo.md(
            "**Figure 1.** Layer × head map of the 40 Atlas task heads (filled blue) and the "
            "lexical task heads of each readout (orange rings). A ring around a blue dot is a "
            "head in both sets. Hover a head for its score $s^h$."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(Counter, atlas_task, decoded, hr, lens_name, lexical, matched, mo, np, per_head, token_text):
    _vocab_cache: dict[int, str] = {}

    def _word(token_id):
        if token_id not in _vocab_cache:
            _raw = token_text(token_id)
            _vocab_cache[token_id] = _raw.strip() or repr(_raw)
        return _vocab_cache[token_id]

    _heads = np.flatnonzero(atlas_task & lexical.any(1))
    _frame = per_head.iloc[_heads].copy()
    for _j, _lens in enumerate(hr.LENSES):
        _words = []
        for _h in _heads:
            _counts = Counter()
            for _top, _hit in zip(decoded["cf_top"][:, _h, _j], matched[:, _h, _j]):
                _counts.update({_word(int(t)) for t in _top[_hit]})
            _words.append(", ".join(w for w, _ in _counts.most_common(6)))
        _frame[f"most frequent task words, {lens_name[_lens]}"] = _words
    shared_heads = _frame.drop(columns=[c for c in _frame if c.startswith("lexical")]).sort_values("Atlas task rank")

    mo.vstack([
        mo.md("""
        ### 3.3 The heads that are lexical and Atlas task heads

        The matched words name the kind of answer: "actor", "singer", "writer" and "person"
        for who-questions, "date" and "year" for when-questions, "location" for
        where-questions (Table 3). Seven in ten items are who-questions, so words for people
        are the most common matches. L16H25, the head that Yang et al. show writing "cities"
        and "companies", is the second-ranked Atlas task head and has the highest score of
        all heads under J-head. The other 32 Atlas task heads, among them several of the
        highest-ranked, match fewer than one item in twenty under both readouts (Table 4).
        """),
        mo.md("**Table 3.** Atlas task heads that are lexical task heads under at least one "
              "readout, with the matched task words they write most often."),
        mo.ui.table(shared_heads.round(3), selection=None, pagination=False, show_column_summaries=False),
        mo.md("**Table 4.** Every head's score under both readouts; sort by any column."),
        mo.ui.table(per_head.round(3), selection=None, page_size=10, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Conclusion

    The lexical-task criterion does not recover the Atlas task heads, nor any other Atlas
    set. A few heads pass both it and the task criterion, L16H25 the clearest among them,
    and under J-head that overlap exceeds what depth alone would give. The high-scoring
    heads also include more parametric heads than chance would, and no more in-context or
    retrieval heads. But most Atlas task heads never write a word for the question's task,
    and most heads that do lie a few layers later, in no Atlas set. The heads whose
    attention to the question matters for the answer and the heads that write what kind
    of answer the question asks for are, for the most part, different heads.

    ## References

    - Achtibat, R., et al. (2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. *ICML*.
    - Kahardipraja, P., Achtibat, R., Wiegand, T., Samek, W., & Lapuschkin, S. (2025). [The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](https://arxiv.org/abs/2505.15807). *NeurIPS*.
    - Merlo, F. (2026). Reading Attention Heads Through the Jacobian Lens.
    - Yang, Z., Li, J. X., Piedrahita Velez, F., Todd, E., Bau, D., Littman, M. L., Bach, S. H., & Pavlick, E. (2026). [Shared Lexical Task Representations Explain Behavioral Variability in LLMs](https://arxiv.org/abs/2604.22027). *ICML*.
    """)
    return


if __name__ == "__main__":
    app.run()
