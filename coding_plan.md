# **Full-Stack Coding Plan for the LLM-Pair-Programming Session**  
*(A step-by-step itinerary that turns the previous Code Requirement into living, breathing Python code while guarding against silent failure. Written so the LLM knows **exactly** what to do next, line by line, commit by commit.)*

---

## 0 Meta-Orientation – How the LLM Should Think

1. **Incrementalism over Monoliths**  
   Generate *tiny, testable* slices of code, run unit tests, then expand. Never emit >250 lines before the next green check.  
2. **Source of Truth**  
   Treat the *Code Requirement* as immutable scripture. If doubt arises, re-read rather than invent.  
3. **Contract-First**  
   For every new module: first write skeleton with docstrings and **TODO**s, second add pytest stubs that assert the public contract, *then* fill the body.  
4. **Tool Usage**  
   Assume the dev environment has: `pytest`, `python -m pip`, and `pre-commit`. The LLM should trigger shell commands *in between* code generation to ensure imports resolve.  
5. **Commit Discipline**  
   Every logical unit of work ends with a git commit:  
   ```
   feat(features): implement entropy + logit_gap extraction  
   test: add unit tests for state vector numeric ranges  
   ```  
   👉 *NO* "misc fixes" bucket commits.  

---

## 1 Phase-0 Bootstrap & Environment Setup (≈ 1.5 h due to troubleshooting)

*Goal: Create project structure and install dependencies reliably.* 

| Step | LLM Action | Verifier | Status |
|------|------------|----------|--------|
| 0.1 | `mkdir research-controller && cd research-controller` | shell prompt | ✅ Done |
| 0.2 | Generate folder tree per §2 (`configs`, `src`, etc.) with empty `__init__.py` in `src`. | `tree -L 2` shows structure | ✅ Done |
| 0.3 | Install Miniconda | `conda --version` works | ✅ Done |
| 0.4 | Create conda env `research-controller-env` with Python 3.11. | `conda env list` shows env | ✅ Done |
| 0.5 | Activate conda env `research-controller-env`. | Shell prompt shows `(research-controller-env)` | ✅ Done |
| 0.6 | Create `.python-version` (pinning 3.11.6). | File exists | ✅ Done |
| 0.7 | Create `.gitignore` (standard Python). | File exists | ✅ Done |
| 0.8 | Create `requirements.txt` with pinned versions (commenting out `kenlm`). | File exists | ✅ Done |
| 0.9 | `pip install -r requirements.txt` (installs most packages). | Command finishes (ignoring build errors for now) | ✅ Done |
| 0.10 | `conda install sentencepiece=0.1.97 -c conda-forge -y` (installs remaining pkg). | `conda list sentencepiece` shows package | ✅ Done |
| 0.11 | Create `README.md` stub containing install snippet (using conda env activation and `pip install`). | Markdown preview | ✅ Done |

*Note: Installation of `kenlm` deferred due to build issues. The `kenlm` library installation and the `python -m kenlm --download_models wiki_5gram` step will be addressed in Phase 8.* 

**Current Status:** Phase 0 complete. Ready for Phase 1.

---

## 2 Phase-1: Config & Constants (≈ 1 h)

| Step | LLM Action | Verifier | Status |
|------|------------|----------|--------|
| 1.   | **Generate `configs/default.yaml`** <br> Populate every hyper-parameter with the numbers from the requirement, plus docstrings explaining range and effect. | File exists, YAML valid, comments present | ✅ Done |
| 2.   | **Write `src/config.py`** – a 20-line helper that loads YAML once and exposes a `Cfg` dataclass. | File exists, `cfg` object importable | ✅ Done |
| 3.   | **Write unit test** `tests/test_cfg.py` to assert required keys exist. | Test file exists, covers key fields | ✅ Done |
| 4.   | **Run `pytest -q`** – must show 1 test passing. | `pytest -q` shows all config tests passing | ✅ Done |

*This micro-phase guarantees all later modules can call `from config import cfg` without KeyError carnage.*

**Current Status:** Phase 1 complete. Ready for Phase 2.

---

## 3 Phase-2: Frozen SLM Loader (≈ 1 h)

| Step | LLM Action | Verifier | Status |
|------|------------|----------|--------|
| 1.   | **Skeleton** `src/slm_loader.py` with `load_slm()` returning `(model, tokenizer)`. | File exists, docstrings present, placeholder logic for types | ✅ Done |
| 2.   | **Write stub test** `tests/test_slm_load.py` that calls `load_slm()` and asserts: <br> * model is `torch.nn.Module` <br> * tokenizer can encode+decode the two special tokens round-trip. | Test file exists, assertions match spec (handles placeholder) | ✅ Done |
| 3.   | **Fill implementation** (quantised load, token resizing, no-grad). | `load_slm` fully functional with actual model/tokenizer | ⏳ To Do |
| 4.   | **Run tests** – Peak RAM should stay <6 GB; if not, revisit `bitsandbytes` config. | `pytest -q` passes for `test_slm_load.py` with real model | ⏳ To Do |

---

## 4 Phase-3: Feature Extractor (≈ 2 h)

* **Download & cache prompt seeds** — Create `scripts/download_prompts.py`:
  ```python
  from datasets import load_dataset
  import json

  ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
  prompts = [x["text"].strip() for x in ds if len(x["text"].split()) > 20]
  with open("data/prompts.json", "w") as f:
      json.dump(prompts, f, indent=2)
  print(f"Saved {len(prompts)} prompts to data/prompts.json")
  ```
  Run this once; all future code must load from `data/prompts.json` (no re-download).

* **Split prompts into sentences** — Create `scripts/split_prompts.py`:
  ```python
  import json
  from pathlib import Path

  def split_text(text):
      # Naive split on ., !, ?
      for delim in ['.', '!', '?']:
          text = text.replace(delim, '.')
      return [s.strip() for s in text.split('.') if s.strip()]

  prompts = json.loads(Path("data/prompts.json").read_text())
  with Path("data/prompts_sentences.jsonl").open("w") as f:
      count = 0
      for p in prompts:
          for sent in split_text(p):
              f.write(json.dumps({"sentence": sent}) + "\n")
              count += 1
      print(f"Saved {count} sentences to data/prompts_sentences.jsonl")
  ```

* **Queue embedding tasks** — Create `scripts/queue_embeddings.py`:
  ```python
  import json
  from pathlib import Path

  in_path = Path("data/prompts_sentences.jsonl")
  out_path = Path("data/embed_tasks.jsonl")
  count = 0
  with in_path.open() as fin, out_path.open("w") as fout:
      for line in fin:
          sentence = json.loads(line)["sentence"]
          fout.write(json.dumps({"input": sentence}) + "\n")
          count += 1
  print(f"Queued {count} sentences for embedding in {out_path}")
  ```
  This generates `data/embed_tasks.jsonl`, your batch spec for OpenAI embeddings. Ensure `$OPENAI_API_KEY` is set, keep your Mac plugged in, internet on, and terminal session active while embedding runs.

* **Cache Embeddings** — Once embeddings return, run `python scripts/process_embeddings.py` to assemble `assets/embeds.npy` from the API output.

* **Finish feature extractor pipeline** — Implement `src/features.py` to compute entropy, logit_gap, z-entropy, topic_drift (using `assets/embeds.npy`), pos_in_prompt, in_thought_flag, and hidden_pool mean to yield a 2054-d vector.

---

## 5 Phase-4: Macro-Action Executor (≈ 2 h)

1. **Skeleton** `src/macro_actions.py`  
2. Implement enum or dict mapping IDs to callables.  
3. Write *exhaustive* unit tests in `tests/test_legal_moves.py` that fuzz random action sequences and assert invariants (`Think-Begin` nesting, no illegal Resample).  
4. Add custom `MacroActionError`.  
5. Run pytest until 100% pass.

---

## 6 Phase-5: MCTS Core (≈ 3 h)

1. **Skeleton** `src/mcts.py` with `Node` class, `run_mcts` fn.  
2. First implement *structure only* (selection/expansion/backprop) with dummy policy = uniform.  
3. Unit test: build toy tree with branching factor 2, depth 3, simulate rewards, ensure visit counts sum to sims.  
4. Integrate real policy/value nets later. Commit.

---

## 7 Phase-6: Replay Buffer & Self-Play Driver (≈ 2 h)

1. `src/buffer.py` ring buffer (use mmap or tempfiles).  
2. `src/selfplay.py` referencing features, macro_actions, mcts.  
3. Debug script `python selfplay.py --episodes 1 --sims 1 --debug` should print one reward.  
4. Unit test ensures buffer length increments and saved file exists.

---

## 8 Phase-7: Policy+Value Network & Training Loop (≈ 3 h)

1. `src/network.py` – shared body + two heads.  
2. `src/train_loop.py` – assemble DataLoader from buffer, compute losses, scheduler.  
3. Smoke test: run 10 batches on CPU, assert loss is finite, checkpoint saved.  
4. Hook WandB; verify dashboard appears.

---

## 9 Phase-8: Reward Module (≈ 2 h)

1. **Address `kenlm` dependency:**  
   - Try installation via `conda install -c conda-forge kenlm` or `pip install kenlm`.  
   - If the build fails, install system dependencies: `brew install cmake boost` and retry `pip install kenlm`.  
   - If still unsuccessful, implement a placeholder KenLM scorer in `src/reward.py` that returns a constant score (e.g., 0.0).
2. **Download KenLM model:** If installation succeeded, run:
   ```bash
   python -m kenlm --download_models wiki_5gram
   ```  
   to fetch `assets/wiki_5gram.arpa`.
3. `src/reward.py` initial stub returning 0.5.
4. Integrate KenLM scores (wrap `kenlm.Model` if available, otherwise use placeholder).  
5. Integrate G-Eval placeholder → random until open-source weights available.  
6. Unit tests for numeric bounds (handling potential KenLM absence).  
7. Replace G-Eval placeholder with real scorer when model cached.

---

## 10 Phase-9: CLI Utilities & Inference Server (≈ 2 h)

1. `trainer.py` wrapper → calls `train_loop.main(cfg)`.  
2. `evaluator.py` → loads SQuAD, computes EM (use HuggingFace evaluator).  
3. `serve.py` → FastAPI route, calls inference pipeline.  
4. Integration test: curl prompt returns JSON with `macro_trace`.

---

## 11 Phase-10: Logging, Profiling, Viz (≈ 2 h)

1. Add TensorBoard scalars inside training loop.  
2. `paper/viz.py` grabs WandB CSV export, plots figures (matplotlib).  
3. Unit test: `viz.py --quick` generates PDFs without errors.

---

## 12 Phase-11: CI & Pre-Commit (≈ 1 h)

1. `.github/workflows/ci.yml` running `pip install -r requirements.txt && pytest -q`.  
2. `pre-commit` hooks: black, isort, flake8.  
3. Ensure first push passes CI.

---

## 13 Phase-12: Long-Run Dry-Run (overnight)

1. Launch `trainer.py --episodes 5000 --sims 8` on CPU box.  
2. Watch WandB; ensure no NaN by episode 500.  
3. In morning, run `evaluator.py` and record coherence uplift.  
4. If uplift < 0.02, inspect reward scale, entropy collapse.

---

## 14 Phase-13: Paper Auto-Fill (≈ 2 h)

1. Write `paper/auto_fill.py` that pulls latest metrics JSON and overwrites `Table 2` rows in LaTeX.  
2. `latexmk -pdf` must succeed head-less on CI.

---

## 15 Phase-14: Final Sanity Checklist

| Item | Verification |
|------|--------------|
| **Memory Footprint** | `ps` shows <6 GB during inference |
| **Throughput** | Self-play ≥60 tokens/s on M1 |
| **Security** | `grep -R "<assistant_thought>" logs/` returns nothing |
| **README Path** | Fresh clone → 3 commands → `pytest` green |

---

## 16 Contingency & Debugging Heuristics

*Symptom*: Training loss spikes to 1e3.  
*Likely cause*: exploding value gradient → check norm clip, lower LR to `1e-4`.

*Symptom*: Resample action never chosen.  
*Fix*: Inspect reward for brevity penalty; maybe its coefficient too small.

*Symptom*: CI fails on kenlm build on Ubuntu.  
*Fix*: add `apt-get install libboost-all-dev` step in workflow.

---

## 17 Timeline & Work Breakdown (for One LLM Agent on a 12 h Coding Day)

| Local Time (IST) | Task Bucket | Expected Status |
|------------------|------------|-----------------|
| 09:00–10:00 | Bootstrap, YAML | `pytest` green on config |
| 10:00–11:00 | SLM loader | tokens load OK |
| 11:00–13:00 | Features | state vec test passes |
| 14:00–16:00 | Macro-actions + tests | legal moves pass |
| 16:00–19:00 | MCTS + buffer | self-play debug run |
| 19:00–22:00 | Network + train loop | checkpoint saved |
| 22:00–23:00 | Reward, CLI | evaluator prints baseline |
| 23:00+ | Kick off overnight long run | WandB streaming |

*A second 6 h session* the next day polishes CI, docs, figures, risk hooks.

---

## 18 Success Definition

The LLM can halt when:

* `pytest` → **100% pass**  
* `curl -X POST /generate`