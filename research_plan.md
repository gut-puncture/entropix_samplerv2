AlphaZero-Style Macro-Action Controller for Text Generation

A fully worked-out research & engineering playbook, written so that a non-coder can still follow every decision and a coder can translate each paragraph straight into functions.

⸻

0 Mental Picture of the System (WHY before WHAT)

Imagine you already have a decent writer—a frozen small language model (SLM).  In normal sampling it picks one word after another with either greedy (always the top word) or temperature (a bit of randomness).  The problem: the SLM sometimes hesitates, sometimes gets over-confident and barrels into nonsense, and sometimes would benefit from pausing to "think".

Goal Wrap that SLM in a coach that, at every word boundary, chooses one of five macro moves:
• Argmax – "you look sure, just say the most likely word."
• Branch – "you're sure but there are a few good options; let's explore the best three continuations and keep the one that leads to the best overall paragraph."
• Resample – "you're confused and keep changing your mind; let's back up to the last punctuation and try a different phrasing."
• Think-Begin / Think-End – "explicitly open an internal notebook page, jot a private chain-of-thought, close it, then continue with a clearer head."
• Temperature-Bump – "just for one token, loosen up a bit so you consider more exotic words."

The coach itself is learned with AlphaZero: it runs a tiny Monte-Carlo tree search (MCTS) in its head, uses two small neural nets (policy π and value v) to guide that search, generates sample texts against itself ("self-play"), receives a reward that measures coherence & smoothness, and gradually improves.  All heavy lifting (vocabulary, grammar, world knowledge) stays in the frozen SLM—so we need no giant GPUs and no huge dataset.

⸻

1 Folder Layout Explained in Plain English

research-controller/
 ├─ configs/            # editable *.yaml files – all knobs in one place
 ├─ data/               # datasets fetched once, safe to delete & re-download
 ├─ assets/             # tokenizer with new tokens, kenlm 5-gram, cached embeds
 ├─ src/                # Python code – every sub-module is explained later
 ├─ checkpoints/        # π_θ and v_θ weights every 2 hours of training
 ├─ logs/               # TensorBoard & Weights-and-Biases run dirs
 └─ paper/              # LaTeX; figures auto-exported from viz.py

If you ever lose track, open configs/default.yaml; its comments mirror the sections below.

⸻

2 Environment Set-Up with Commentary

# 1. install Python 3.11 via pyenv
# 2. create venv & install libs
# 3. kenlm compile helper (first time only)
# 4. set your OpenAI key so topic-drift embeds work

Note: We need bitsandbytes. Allows 4-bit quantisation of the SLM → fits in 5 GB RAM.

⸻

3 Frozen Small Language Model (SLM)
	•	Model chosen: Phi-3-Mini-128k-Instruct (3.8 B). It'll run on 8 GB MacBook Air M1.
	•	Context length: 128 k tokens – future-proofed for long prompts but we'll actually cap generation at 512 tokens in experiments.

Adding private-thought tokens step-by-step

SPECIAL_TOKENS = ["<assistant_thought>", "</assistant_thought>"]

Hiding of internal thoughts will be done by our stripper function after generation. We won't show anything inside the thinking tags to the user.

⸻

4 Datasets with Rationale and Download Snippets

4.1 Prompt seeds (self-play start contexts) – WikiText-2 because it is clean English, only 4.5 MB, and contains multi-sentence paragraphs (needed for topic-drift calculations).

Sample Code (may need modification):
from datasets import load_dataset
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
prompts = [x["text"].strip() for x in wikitext if len(x["text"].split())>20]

4.2 Topic-drift embeddings – expensive if we call OpenAI every step. So we pre-embed every sentence in the seed prompts once and write them to assets/embeds.npy.

from openai import OpenAI
client = OpenAI()
vecs=[]
for sent in sentences:  # break by spaCy or simple period split
    resp = client.embeddings.create(model="text-embedding-3-large", input=sent)
    vecs.append(resp.data[0].embedding)
np.save("assets/embeds.npy", np.asarray(vecs, dtype=np.float32))

4.3 QA eval sets – SQuAD & BoolQ for "did you answer correctly?" sanity.  Not used in training.

⸻

5 Feature Extractor — Detailed Narrative + Pseudocode

At every real or simulated step we need a fixed-length numeric snapshot of the current context.  This goes into π_θ and v_θ.

Components explained
	•	entropy – how spread out are next-token probabilities.  High entropy → the SLM is unsure.  Compute after soft-max because logits scale can drift.
	•	logit_gap – margin between the best and second-best candidate.  Complements entropy (which averages over the whole vocab).
	•	z_entropy – entropy normalised inside a sliding window of the last 16 tokens so the network sees relative spikes not absolute scale.
	•	topic_drift – semantic jump from previous sentence to current.  0 = on the same topic, 1 = huge change.  Gives the policy a clue when a new paragraph start is imminent.
	•	pos_in_prompt – how far we are into the 128 k context.  Near the end we may want to rush and avoid Branch.
	•	in_thought_flag – 1 if inside <assistant_thought> … </assistant_thought> pair, because Think-End becomes a mandatory legal move.
	•	context_vec – the representation of the entirety of the tokens generated so far. pooled last hidden layer (mean over sequence).  Provides rich info like tense, subject, etc. in 2048 numbers.

Pseudocode (Python-style comments). Please be careful while using this. It may be outdated, with wrong syntax, might not follow the requirements etc.

def extract_state(generator, tokenizer, cache, tokens, hidden_pool, feature_hist):
    # 1. entropy + logit_gap
    with torch.no_grad():
        logits = generator.forward(tokens, use_cache=True, past_key_values=cache).logits[:,-1,:]
    probs  = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
    top2    = torch.topk(logits, k=2, dim=-1).values.squeeze()
    logit_gap = (top2[0] - top2[1]).item()

    # 2. z-entropy (update rolling window)
    feature_hist.append(entropy)
    if len(feature_hist) > 16:
        feature_hist.pop(0)
    mu = np.mean(feature_hist); sigma = np.std(feature_hist)+1e-5
    z_entropy = (entropy - mu)/sigma

    # 3. topic drift (only at sentence boundary)
    if tokens[-1] in tokenizer.convert_tokens_to_ids(["."]):
        prev, curr = last_two_sentence_embeds()
        topic_drift = 1 - cosine(prev, curr)
    else: topic_drift = 0.0

    # 4. misc
    pos_in_prompt = len(tokens)/generator.config.max_position_embeddings
    state_vec = [entropy, logit_gap, z_entropy, topic_drift, pos_in_prompt,
                 in_thought_flag]
    state_vec.extend(hidden_pool.astype(np.float16).tolist())  # now 2054 dims
    return torch.tensor(state_vec)

hidden_pool is produced by registering a forward-hook on the final transformer layer and averaging activations.

⸻

6 Macro-Action Execution with Inline Commentary

Exercise caution while using the code:
def apply_macro_action(action_id, …):
    if action_id==0:  # Argmax
        token = probs.argmax(); append(token)
    elif action_id==1:  # Branch
        topk = torch.topk(probs, k=3).indices
        child_states = []
        for tok in topk:
            Δh = capture_delta_kv(tok)   # returns tuple of k,v for each layer
            child_states.append((tok, Δh))
        return child_states  # tree expansion happens in mcts.py
    elif action_id==2:  # Resample
        back_idx = find_prev_punct(tokens)
        tokens = tokens[:back_idx]
    elif action_id==3:  # Think-Begin
        append(tokenizer.convert_tokens_to_ids("<assistant_thought>"))
        in_thought_flag = 1
    elif action_id==4:  # Think-End
        append(tokenizer.convert_tokens_to_ids("</assistant_thought>"))
        in_thought_flag = 0
    elif action_id==5:  # Temp-Bump
        logits *= 1.3
        token = sample_from(logits)
        append(token)

Every branch stores only (token_id, Δh); full cache is reconstructed on demand: torch.cat([parent_k, Δh.k], dim=-2).

⸻

7 MCTS Inner Loop — Thorough Walkthrough

Exercise caution while using the code:
def run_mcts(root_state, sims=8):
    for _ in range(sims):
        path = []
        node = root_state

        # 1. SELECTION – follow UCB
        while node.is_expanded():
            action, node = node.select_child(c=1.5)  # returns edge + new node
            path.append((node, action))

        # 2. EXPANSION – pick all legal actions at leaf
        node.expand_with_priors(policy_net(node.state_vec))

        # 3. ROLLOUT – quick plagiarism-safe rollout
        reward = rollout_simulation(node, max_len=32)

        # 4. BACK-PROP
        for node, action in path:
            node.update_stats(reward)

    # 5. RETURN policy over root edges
    visits = torch.tensor([child.N for child in root_state.children])
    π_mcts = visits / visits.sum()
    return π_mcts

Why only 8 sims?  Each sim includes at most three extra SLM forwards (for Branch) → ≈200 ms on CPU.  Raising to 16 doubles decision quality but nearly doubles latency; we do that only after epoch 10 when π_θ is better.

⸻

8 Self-Play Loop End-to-End with Motivation Comments

Exercise caution while using the code:
def play_episode(generator, π, v, buffer):
    prompt = random.choice(prompts)[:64]  # keep it short so we see many turns
    state = reset_state(prompt)

    traj = []
    while not (state.eos or len(state.tokens)>256):
        π_mcts = run_mcts(state)          # *search* improves raw π
        action = π_mcts.argmax()          # we act greedily during training
        state = apply_macro_action(action)
        traj.append((state.snapshot(), π_mcts))

    R = reward(strip_thought(state.tokens))  # coherence etc.
    for s,πm in traj:
        buffer.append((s, πm, R))

Why store one R for all moves?  It aligns with AlphaZero's idea that each position eventually leads to the same game outcome; here the "game" is the final paragraph.

⸻

9 Training Step Explained Top-to-Bottom

Exercise caution while using the code:
def optimise(batch):
    state_vecs = torch.stack([b[0] for b in batch])
    target_policy = torch.stack([b[1] for b in batch]).float()
    target_R = torch.tensor([b[2] for b in batch]).float()

    pred_policy_logits, pred_value = network(state_vecs)
    pred_value = pred_value.squeeze()

    loss_value  = F.mse_loss(pred_value, target_R)
    loss_policy = -(target_policy * F.log_softmax(pred_policy_logits,-1)).sum(-1).mean()
    loss_ent    = -0.01 * (F.softmax(pred_policy_logits,-1)*
                            F.log_softmax(pred_policy_logits,-1)).sum(-1).mean()
    loss = loss_value + loss_policy + loss_ent

    loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(),1.0)
    optimiser.step(); optimiser.zero_grad()

Intuition – value MSE teaches the net to predict eventual coherence; policy CE distils the improved search policy; entropy bonus prevents over-confident π.

⸻

10 Live Monitoring with Real-World Interpretations
	•	wandb line charts – training loss should stabilise then very slowly go down; sudden spikes → numerical blow-up (check grads).
	•	histogram of visit counts – healthy controller sees each action ≥ 5 % after 5 epochs; if Argmax dominates 95 % you may have mis-weighted reward.
	•	coherence vs epoch – expect +0.02–0.04 absolute improvement by epoch 20.
	•	speed gauge – target ≥60 self-play tokens/s on CPU; if slower run viz.profile() to detect hot spots (often JSON logging).

⸻

11 Before Committing a 24-h Run (Sanity Tests)
	1.	pytest tests/test_deltah.py — create parent + Δh reconstruction and assert equality.
	2.	pytest tests/test_legal_moves.py — random sequences of Think-Begin/End ensuring flag toggles.
	3.	python trainer.py --episodes 10 --sims 1 --debug — prints state vector min/max; check none are nan.
	4.	python evaluator.py --ckpt checkpoints/debug.pt — expect coherence 0.40±0.05 (baseline greedy ≈0.35).
	5.	If all good, kick off:

python trainer.py --episodes 50000 --sims 8 --save_every 7200 &



⸻

12 After Training — Inference API Walkthrough

A user prompt comes in:
	1.	Initialise state with prompt tokens; feature extractor fills vector.
	2.	While not EOS & tokens<256:
a. run run_mcts (sims=16) → π.
b. if --stochastic sample from π; else argmax.
	3.	Post-process tokens: delete <assistant_thought> block and collapse repeated whitespace.
	4.	Return visible text.

⸻

13 Paper Section-by-Section Checklist with Data Sources
	•	Introduction – cite Welleck 2020 (self-bLEU), G-Eval 2024 for human-style eval.
	•	Method – include Algorithm 1 (pseudo identical to §8).  Figure 1: colour-coded pipeline boxes.
	•	Experiments – Table 2 includes greedy, nucleus 0.95, our controller (8 sims) and ablation (no Branch).  Report coherence, QA EM and tokens/sec.
	•	Discussion – talk about compute-cost vs benefit (controller adds 1.8× latency but +0.03 coherence).
	•	Appendix – full hyper-param YAML, reward weight sweep plot.

⸻

14 Risk Matrix Expanded

Risk	Example	Likelihood	Impact	Response
reward mis-scales	Coh in [0,1], ΔH sum ≈10 ⇒ length dominates	Med	Med	normalise each term to [0,1] first
thought leakage	user sees "<assistant_thought> I'm stuck >"	Low	High PR	regex strip + unit test
OpenAI outage	embeddings API down	Med	Low	fall back to cached last δ



⸻

15 Future Work (beyond first paper)
	•	Plug-in RL-HF style preference model instead of static coherence.
	•	Increase search depth to 2 macro-actions; re-use value network as heuristic so cost stays sub-quadratic.
	•	Port to Metal-optimized PyTorch on M2 for 4× speed.

⸻

END — You Can Hand This to an Engineer

Every paragraph links back to an explicit code module or experiment decision.  Follow sections sequentially: set-up → tiny tests → long run → paper.  If anything is unclear, highlight the passage in this canvas and we'll iterate.