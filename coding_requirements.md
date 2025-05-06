# **Code Requirement**  
*AlphaZero-Style Macro-Action Controller for Text Generation*  
*(Everything a developer and a reviewer need—no tacit knowledge left between the lines)*  

---

## 0  Purpose and Core Idea (Why We Are Building This)  
We already possess a perfectly competent *small* language model (SLM) that knows English, grammar, and world facts—but it is oblivious to *situational self-control*.  On hard passages it rambles or stalls; on easy ones it wastes compute arguing with itself.  We therefore bolt a *learned* "coach" on top that, **at every token boundary**, decides which *macro-action* the frozen SLM must execute next.  The coach is trained with the AlphaZero recipe—Monte-Carlo Tree Search (MCTS) guided by policy π and value v networks, refined through self-play on plain English paragraphs.  All heavyweight knowledge lives in the frozen SLM; the coach merely learns *when to branch, when to think, and when to chill*.  The result is higher-coherence prose at tiny extra compute cost, and a research story suitable for a systems-and-RL conference paper.

---

## 1  Scope of Work (What Must Exist When We Ship)  
By the end of the sprint the repo **must contain**:  

1. **Reproducible environment scripts**—shell commands that turn a blank macOS/Linux box with ≥8 GB RAM into a runnable dev setup in <20 min.  
2. **Folder skeleton** already populated with placeholder files and exhaustive inline comments so a newcomer can open any `*.py` and see *where* to fill code without guessing intent.  
3. **Quantised frozen SLM load** (`Phi-3-Mini-128k-Instruct` in 4-bit) plus tokenizer patching for private-thought tokens.  
4. **Standalone feature-extractor module** that, given a running SLM KV-cache and recent token history, returns a *2054-dim float32* state vector **exactly** as documented.  
5. **Macro-action executor** implementing six legal moves (`Argmax … Temp-Bump`) with complete state bookkeeping (token list, KV-delta stack, `in_thought_flag`).  It must guarantee that `Think-Begin`/`Think-End` remain properly nested—violations throw an exception so bugs surface instantly.  
6. **General MCTS engine** instrumented for our mixed discrete/continuous domain:
   * node statistics `N`, `W`, `Q`, `P` persisted in FP32;  
   * UCB formula `U = c·P·√(N_parent)/(1+N)` with default `c=1.5`;  
   * pluggable *rollout* callback (we will initially use a cheap 32-token greedy rollout).  
7. **Self-play driver** that streams (`state_vec`, `π_mcts`, `R`) tuples into a *disk-backed* replay buffer (use a plain-pickle ring set capped at 500 k transitions ≈1 GB).  
8. **Unified policy-and-value network**:
   * input layer size = 2054;  
   * one hidden block of `nn.Linear(2054→2048)`, GELU, LayerNorm;  
   * **shared** body followed by two heads:  
     * policy head `2048→128→6` (logits);  
     * value head  `2048→128→1` (scalar).  
   Initialise with Xavier-Uniform; value head final bias = 0.  Store checkpoints every 2 h.  
9. **Training loop** with (i) cited loss formula `loss = MSE + CE + λ·entropy`, (ii) AdamW (`lr=3e-4`, `betas=(0.9,0.99)`, `wd=1e-2`), (iii) gradient clipping at 1.0.  
10. **Reward module** exposing a single `score(text:str)->float` that blends:  
    * `+G-Eval-coherence` *z-scored* to [0,1],  
    * `−0.2·Δtopic_drift`,  
    * `+0.1·KenLM_logprob_per_token` *min-maxed* to [0,1],  
    * `−0.002·|tokens-256|` (length penalty centred on 256).  
    Unit tests must prove each component outputs finite values for a 3-sentence dummy paragraph.  
11. **CLI utilities**:  
    * `trainer.py` – launches self-play + optimisation;  
    * `evaluator.py` – loads a checkpoint, runs on SQuAD/BoolQ, prints EM & coherence;  
    * `serve.py` – lightweight Flask or FastAPI endpoint implementing §12 inference flow, returning JSON `{"text":…, "macro_trace":[…]}`.  
12. **Experiment logging** wired to *both* TensorBoard *and* Weights-and-Biases, capturing:
    * total env-steps, self-play tokens/sec, π visit histograms, loss curves, reward moving average.  
13. **Test suite** (pytest) described in §11 of the design: Δ-KV reconstruction, legal-move toggling, NaN guards, baseline coherence sanity.  
14. **LaTeX `paper/` directory** containing template sections, figure placeholders, and a Python `viz.py` that auto-exports:  
    * Figure 1 pipeline diagram,  
    * Training curves,  
    * Ablation table.  
15. **Risk-mitigation hooks**:  
    * regex strip of `<assistant_thought>` blocks before display;  
    * API-key presence check with graceful fallback to cached embeddings;  
    * config YAML-level normalisation of every reward term to protect against unit drift.  
16. **Comprehensive `README.md`** that glues everything above into a narrative: install → sanity-tests → 24 h run → paper figure regeneration.  It ends with an "If something explodes" FAQ listing every known failure mode and its quick fix.

---

## 2  Folder Layout (Absolute Truth Source)  
```
research-controller/
├─ configs/
│  └─ default.yaml            # every hyper-param + docstring
├─ data/                      # fetched datasets (.jsonl, .npy, .arpa)
├─ assets/
│  ├─ tokenizer_extended/     # tokenizer.json + added tokens file
│  └─ wiki_5gram.arpa         # KenLM model auto-downloaded
├─ src/
│  ├─ slm_loader.py
│  ├─ features.py
│  ├─ macro_actions.py
│  ├─ mcts.py
│  ├─ selfplay.py
│  ├─ network.py
│  ├─ reward.py
│  ├─ buffer.py
│  └─ train_loop.py
├─ checkpoints/
│  └─ πv_epoch000.pt
├─ logs/                      # TB & wandb sub-dirs
├─ tests/
│  ├─ test_deltah.py
│  ├─ test_legal_moves.py
│  └─ test_debug_run.py
├─ scripts/                   # one-off dataset grabbers
├─ paper/
│  ├─ main.tex
│  ├─ figures/
│  └─ viz.py
└─ README.md
```
Every file's first 15 lines *must* explain its responsibility, imports, and cross-module contract so grep-only navigation suffices.

---

## 3  Environment & Dependency Contract  
*Platform guarantee*: macOS 15.4 Seqouia, Python 3.11, ≤8 GB RAM, no dedicated GPU.  
*Proposed install sequence* (just a guideline):

```bash
brew install pyenv                # or apt-get python-build dependencies
pyenv install 3.11.6 && pyenv local 3.11.6

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# Install all required packages with pinned versions
pip install -r requirements.txt

python -m kenlm --download_models wiki_5gram
export OPENAI_API_KEY="sk-<YOUR-KEY>"
```
The `requirements.txt` file ***must*** pin exact versions so hashes remain deterministic for artifact reproducibility. See `requirements.txt` for the full list.

---

## 4  Frozen SLM Specification  
*Model*: **`microsoft/Phi-3-Mini-128k-Instruct`** (3.8 B params).  
*Load mode*: 4-bit GPTQ via `bitsandbytes`; expect ~5 GB CPU RAM footprint.  
*Tokenizer patch*: append `"<assistant_thought>"` and `"</assistant_thought>"` once, then `slm.resize_token_embeddings(len(tok))`.  Persist extended tokenizer to `assets/tokenizer_extended/` so training and inference agree bit-for-bit.  

*Guarantee*: Loader returns `(model, tokenizer)` in eval mode with gradient disabled; feature extractor re-uses the single global instance to avoid memory bloat.

---

## 5  Macro-Action Catalogue (Official Legal Move Set)  

| ID | Name              | One-Liner Effect                                                     | Extra State Mutation                                  |
|----|-------------------|----------------------------------------------------------------------|-------------------------------------------------------|
| 0  | **Argmax**        | Append most-likely token.                                            | —                                                     |
| 1  | **Branch**        | Save top-3 continuations & KV-deltas, feed to MCTS child nodes.      | Push `(tok, Δh)` per child into tree.                |
| 2  | **Resample**      | Roll back to last comma/period, clear cache from that point.         | Trim `tokens`, pop KV-stack accordingly.              |
| 3  | **Think-Begin**   | Insert `<assistant_thought>` token, set `in_thought_flag = 1`.       | —                                                     |
| 4  | **Think-End**     | Insert `</assistant_thought>` token, set `in_thought_flag = 0`.      | — (illegal unless flag==1)                            |
| 5  | **Temp-Bump**     | Multiply logits by 1.3 then sample *once* with temperature = 1.      | —                                                     |

*Invariants enforced*:  
- `Think-Begin` *must* precede `Think-End`; nesting depth strictly 0/1.  
- `Resample` forbidden if no punctuation exists before current token.  
- `Branch` allowed only when `entropy < 2.0 ∧ logit_gap > 1.0` else wasted compute.  

Violations raise `MacroActionError`, caught in unit tests and logged during training.

---

## 6  State Vector (Feature Extractor) Contract  
Returns `torch.tensor(shape=(2054,))` dtype `float32`.  Concatenation order:  

```
[ entropy,
  logit_gap,
  z_entropy,
  topic_drift,
  pos_in_prompt,
  in_thought_flag,
  pooled_hidden_0, pooled_hidden_1, … pooled_hidden_2047 ]
```

*Implementation details*—entropy from soft-max, logit_gap top-2 margin, z-entropy via rolling window length 16, topic_drift by cosine on pre-cached sentence embeddings, pooling = mean over final hidden layer activations.  All ops wrapped in `torch.no_grad()`; extractor holds an internal `feature_hist` deque to compute z-scores.

---

## 7  MCTS Details (Selection→Expansion→Rollout→Backprop)  
- **Simulations per move**: 8 during epochs 0-9, 16 afterwards (configurable).  
- **UCB constant** `c=1.5`, stored in YAML.  
- **Rollout policy**: greedy sampling for ≤32 tokens; stops on newline or EOS.  
- **Reward propagation**: single terminal reward `R` assigned to *all* states in episode trajectory (`AlphaZero` convention).  
- **Visit-count temperature**: *training* acts greedily (`argmax(visits)`); *inference* may optionally sample (`τ` flag via CLI).  

---

## 8  Self-Play and Replay Buffer  
- Episode truncates at `min(EOS, 256 tokens)`.  
- Replay buffer on-disk ring (`buffer.py`) with *write-through* pickle so crash during long runs does not wipe data.  
- Sampling for training is uniform; future work may add PER.  

---

## 9  Optimisation Loop  
- **Batch size**: 256 state tuples.  
- **Loss**: `MSE(value) + CE(policy) + 0.01·entropy`, where entropy bonus uses predicted π to avoid collapse.  
- **Optimiser**: AdamW (`lr=3e-4`).  Learning rate halved every 200 k optimiser steps via `torch.optim.lr_scheduler.StepLR`.  
- **Grad clip**: `torch.nn.utils.clip_grad_norm_` at 1.0.  
- **Checkpoint** every 2 h wall clock with `state_dict`, optimiser state, replay cursor.  

---

## 10  Reward Calculus (Exact Numbers)  

```
coh   = (g_eval - μ_coh)/σ_coh ∈ [0,1] after pre-computed dataset stats
topic = topic_drift            ∈ [0,1]
kenlm = (lm_per_tok - μ_lm)/(σ_lm) then min-max to [0,1]

R = +1.0*coh  -0.2*topic  +1.0*kenlm  -0.002*abs(len-256)
```

`μ`, `σ` baked into `configs/default.yaml` for determinism.  Unit tests confirm result lies within [-1.5, +1.5] for dummy text.

---

## 11  Logging & Monitoring Targets  
- **wandb.run** tracks `loss_policy`, `loss_value`, `entropy_bonus`, `reward_mean`, `selfplay_toks/sec`.  
- Histogram of macro-action frequencies plotted per epoch.  
- TensorBoard mirrors scalar logs (for offline runs).  
- `viz.py` regenerates these graphs into `paper/figures/` as PDF/PNG for LaTeX.  

---

## 12  Inference API Contract  
HTTP POST `/generate` with JSON `{"prompt": "...", "stochastic": true|false}`  
returns  
```json
{
  "text": "visible answer without thoughts",
  "macro_trace": [
     {"pos":17,"action":"Think-Begin"},
     {"pos":33,"action":"Argmax"},
     …
  ]
}
```  
Latency target: ≤3 s for 256 tokens on M1 CPU with `sims=16`.

---

## 13  Testing and Quality Gates  
| Stage | Script | Pass Condition |
|-------|--------|----------------|
| Delta-KV | `pytest tests/test_deltah.py` | `torch.allclose(parent, reconstruct(parent,Δh))` |
| Macro legality | `pytest tests/test_legal_moves.py` | No `MacroActionError` on 10 k random sequences |
| NaN guard | `pytest tests/test_debug_run.py` | Feature min/max finite, losses finite |
| Baseline eval | `python evaluator.py --ckpt checkpoints/debug.pt` | Coherence ≥0.40 |

CI pipeline on GitHub Actions *must* fail-fast if any gate breaks.

---

## 14  Risk Controls Embedded in Code  
1. **Reward Mis-scale**: normalisation helper in `reward.py`; asserts each term ∈ [0,1].  
2. **Thought Leakage**: regex scrubber in `serve.py`; unit test ensures no `<assistant_thought>` survives.  
3. **OpenAI Outage**: if embeddings call throws, loader falls back to on-disk `.npy`; warning logged.  
4. **Replay Corruption**: buffer writes atomic via `tempfile.NamedTemporaryFile` + rename.  

---

## 15  Documentation & Paper Hooks  
Every algorithmic paragraph in this requirement **maps one-to-one** to a subsection in `paper/main.tex`.  `README.md` lists the mapping so a reviewer can cross-check code ↔ manuscript.  `viz.py` exports:  

- Figure 1: System overview with colour-coded SLM vs Controller vs Data.  
- Table 2: Metrics for baseline vs controller vs ablations.  
- Appendix: YAML dump of all hyper-params and sweep results.

---

## 16  Deliverables & Acceptance Criteria  
*The project is "done" when*:  

- New dev can `git clone`, run **three commands** from README, and reproduce the debug run without manual tweaks.  
- All tests green; CI badge visible.  
- `serve.py` returns coherent answers demonstrably *better* than greedy sampling (≥+0.025 G-Eval).  
- `paper/` compiles under `latexmk` into a PDF that already plots Figure 1 and contains placeholder results tables.  

Once these checkpoints are met we move to the *Coding Plan* (Kanban breakdown, timeline, owner matrix, contingency buffer).  

**END OF CODE REQUIREMENT**