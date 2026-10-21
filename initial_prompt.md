Reproduce the head identification from "The Atlas of In-Context Learning" (https://arxiv.org/abs/2505.15807) on `meta-llama/Llama-3.1-8B-Instruct`, then build a marimo notebook to explore those heads.

**Sources (use, don't reimplement)**
- Official code: https://github.com/pkhdipraja/in-context-atlas. Follow its README pipeline on NQ-Swap to identify the top-100 in-context and parametric heads (§5) and the top-40 task and retrieval heads (§6.1).
- Lenses: https://github.com/filippo-merlo/head_level_jlens. Use its head JLens and logit-head implementations as-is. If you can't access it, stop and ask me.

**Scope**
Stop at head identification. No ablations, function vectors, needle-in-a-haystack experiment or probes.

**Deliverables**
1. `results/heads.json` with each head's (layer, head), category and scores. Also a layer × head map of the categories, and a short comparison against the paper: baseline recall from Table 4, and the layer pattern in Fig. 1.
2. `explore_heads.py` (marimo), containing:
   - Pick a prompt (free text or an NQ-Swap example under closed-book, oracle or counterfactual), a head category, a head and a token position. Show top-k tokens from head JLens and from logit head side by side, plus the head's attention pattern.
   - **Most common decoded words per head category**: over a set of NQ-Swap examples, aggregate the top-k tokens decoded from each category's heads (in-context, parametric, task, retrieval). Show the most frequent ones per category, for each lens and each condition.

**Rules**
- Never invent numbers or round them toward the paper's values.
- Tell me before changing the method because of compute limits or broken code.