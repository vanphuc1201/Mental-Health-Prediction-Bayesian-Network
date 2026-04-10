# ClearMind: Implementation Plan — Stress Detection via Multi-Adapter Feature Extraction and Bayesian Inference

## 1. Project Scope (Narrowed)

The original proposal covered general mental health support across multiple conditions. Based on feedback, this implementation focuses exclusively on **psychological stress detection**. The system remains a neuro-symbolic hybrid:

- **Component A** — Multiple LoRA-fine-tuned adapters (sharing a single base LLM) that each extract a distinct family of stress-related features from free-form text, plus a pure-compute module for lexically trivial features.
- **Component B** — A Bayesian Network that synthesises those features into a calibrated posterior probability of the user's perceived stress level.

The POMDP dialogue loop from the original proposal is deferred; inference is performed on a per-text (or per-session) basis rather than as an interactive conversation.

---

## 2. Architecture Overview

```
┌──────────────┐
│  User Text    │
│  (Reddit /    │
│   free-form)  │
└──────┬───────┘
       │
       ├──── Pure compute (regex) ──────────────────────────┐
       │       → first_person_singular_density              │
       │                                                    │
       ├──── Lexical adapter (LoRA) ────────────────────┐   │
       │       → absolutist, neg_emo, discrepancy,      │   │
       │         hedging, negation_sentiment (5)         │   │
       │                                                 │   │
       ├──── Cognitive adapter (LoRA) ──────────────┐    │   │     ┌──────────────────┐
       │       → 10 cognitive distortions            │    │   │     │  Bayesian Network │
       │                                             │    │   ├────▶│  (pgmpy)          │
       ├──── Affect adapter (LoRA) ──────────┐       │    │   │     │                   │
       │       → anxiety, anger, sadness,    │       │    │   │     │  Stress Level     │
       │         anhedonia, overwhelm,       │       │    │   │     │  → P(not|stressed)│
       │         positive_affect (6)         ├───────┴────┴───┘     └──────────────────┘
       │                                     │
       ├──── Behavioural adapter (LoRA) ─────┤     29 features
       │       → sleep, social, exhaustion,  │     merged into
       │         work, impairment,           │     virtual evidence
       │         help_seeking (6)            │
       │                                     │
       └──── Repetitive thought adapter ─────┘
               (LoRA) → repetitive_thought_density (1)
```

**Key design decision: separate adapters per feature group.** Each LoRA adapter is trained on a dedicated dataset with a focused system prompt, ensuring:
1. **Higher annotation quality** — each group's training data uses the tier-appropriate annotation strategy (compute, cross-map, or LLM consensus).
2. **Modular evaluation** — each adapter can be evaluated, ablated, and retrained independently.
3. **Adapter swapping** — at inference the base model is loaded once; adapters are swapped in/out per group, keeping memory constant.
4. **Interpretability** — the BN can attribute posterior shifts to specific adapter outputs.

---

## 3. Phase 1 — Data Acquisition, Tiered Annotation, and Per-Group Dataset Construction

### 3.1 Datasets

| Dataset | Source | Primary role | Size |
|---|---|---|---|
| **SenticNet Reddit_Combi** | SenticNet stress dataset | **Primary BN training & evaluation** — stressed Reddit posts with binary stress labels, supplemented with EmpatheticDialogues as the negative class. **Not used for adapter SFT.** | 2,745 stressed + supplemented negatives (~5,490 total) |
| **EmpatheticDialogues** | Facebook Research | **BN negative class supplement** — genuinely non-clinical text (grateful, joyful, proud, excited contexts). Solves the contaminated control group problem where "non-stressed" Reddit posts still contained mental-health language. | Supplemented to balance classes |
| **GoEmotions** | [Google/HuggingFace](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Cross-mapping → affect adapter training data. 28 emotion labels mapped to 6 affect features. | 54,263 comments |
| **PatternReframe** | Maddela et al. 2023; [GitHub](https://github.com/SALT-NLP/positive-frames) | Cross-mapping → cognitive adapter. Reframing strategies mapped to 10 distortion labels. | 8,349 examples |
| **TherapistQA** | Shreevastava & Foltz 2021 | Cross-mapping → cognitive adapter. 10 CD labels. | 2,530 examples |
| **Thinking Trap** | Sharma et al. 2023 | Cross-mapping → cognitive adapter. Expert-annotated cognitive distortions, 13 labels mapped to 10-feature schema. | 600 examples |
| **Reddit Mental Health** | [Zenodo](https://zenodo.org/records/3941387) (Low et al. 2020; [JMIR](https://doi.org/10.2196/22635)) | **Primary text source for adapter SFT** — scale, subreddit diversity. LIWC and TF-IDF features in the CSVs are used as **weak signals in frontier model annotation prompts only** (not in SFT data). Posts from 28 subreddits, 2018–2020. PDDL-1.0. | ~3.1 GB; 800K+ unique users |

**SFT / BN firewall:** Adapter training data (`sft_*_train.jsonl` / `val`) is built from Reddit Mental Health (Zenodo), GoEmotions, PatternReframe, TherapistQA, and ThinkingTrap. It **excludes** all BN evaluation texts (SenticNet). Adapters are trained on SFT-eligible corpora, then applied to the SenticNet dataset for BN supervision and evaluation.

**Why SenticNet over Dreaddit:** Dreaddit’s “non-stressed” class comes from mental-health subreddits where even control posts contain clinical language, creating a contaminated negative class. SenticNet Reddit_Combi provides genuinely stressed Reddit posts, while EmpatheticDialogues provides genuinely non-clinical text as the negative class. This was the single biggest driver of BN performance.

### 3.2 Annotation Strategy

All 28 adapter-extracted features are **LLM-annotated** (via frontier model or cross-mapped from existing labelled datasets). The only non-LLM feature is `first_person_singular_density`, which is computed via regex. The annotation strategy varies by feature group based on data availability:

- **Cross-mapped features:** Where an existing labelled dataset covers the target features, labels are cross-mapped directly (affect → GoEmotions, cognitive distortions → PatternReframe/TherapistQA/ThinkingTrap)
- **Frontier LLM-annotated features:** Where no proxy dataset exists, a frontier model (GPT-4o) annotates Reddit Mental Health texts. LIWC/TF-IDF weak signals from the source CSV are included in the prompt to ground the annotation, but are **never** included in the SFT training data
- **Hybrid features:** Some groups combine both strategies (affect uses GoEmotions cross-map + LLM annotation for features without a proxy)

#### Tier 1A — Pure Compute (No LLM)

`first_person_singular_density` is exhaustively definable — the word set is closed (`{i, me, my, mine, myself}`), context adds nothing, and LLM inference would introduce noise not signal. LIWC-validated for decades (Rude et al. 2004). Computed via regex in `train_bayesian_network.py::compute_first_person_singular_density()`.

`first_person_singular_density` also serves as a calibration ablation: compare the LLM's density estimate against the programmatic count.

#### Tier 1A-ext — LLM-Annotated Repetitive Thought (Separate Adapter)

`repetitive_thought_density` was originally planned as pure compute but is instead trained as a **separate 5th LoRA adapter** via Tier 3 frontier LLM annotation on Reddit Mental Health posts. This measures intra-post ruminative cycling — whether the same ideas recur across sentences. It replaces the former rumination adapter group whose 3 features (repetitive thought, brooding vs. reflection, lack of resolution) had no proxy dataset. `brooding_vs_reflection` is dropped (low IAA even among trained clinical raters); `lack_of_resolution` is subsumed by `discrepancy_language` (lexical) and `help_seeking` (behavioural, inverse signal).

Frontier model prompts include LIWC weak signals (`liwc_cognitive`, `liwc_insight`) as contextual grounding, but these are **not** passed through to the SFT dataset — the trained adapter sees only raw text and must learn to predict the score without LIWC.

#### Tier 1B — Multi-Run LLM Span Extraction (Lexical Adapter)

Five lexical features require contextual understanding that a fixed wordlist cannot provide:

| Feature | Why LLM over rules | Research backing |
|---|---|---|
| `absolutist_word_density` | Phrasal absolutism ("zero chance"), context disambiguation ("I absolutely love this" ≠ distress), implicit absolutism ("why do I even bother") | Al-Mosaiwi & Johnstone 2018; d > 3.14 across affective disorders |
| `negative_emotion_density` | VADER fails on multi-clause negation, sarcasm, implicit affect ("I've been... fine") | LIWC/Pennebaker 2010; DASS study 2025 confirms fixed lists → misclassification |
| `discrepancy_language` | LIWC `Discrep` misses "I was supposed to", "if only I had", implicit unmet expectations | Eichstaedt et al. 2018 Facebook depression prediction study |
| `hedging_tentativeness` | LIWC `Tentat` misses stressed hedging: "I guess I'm fine", "I don't know, maybe it's me" | LIWC Tentat category; LLM for social pressure register |
| `negation_aware_negative_sentiment` | VADER fails on "I don't feel terrible today"; multi-clause negation needs semantic parsing | DASS study 2025 |

Each feature uses **multi-run consensus**: run extraction N times (default 5) at temperature 0.35, use run-agreement as the confidence score. High-variance features (stdev > 0.25 across runs) are flagged as genuinely ambiguous.

#### Tier 2 — Cross-Map from Existing Dataset Labels

For features where a source dataset's labels directly correspond to a feature node:

**GoEmotions → `affect` adapter:**

```python
AFFECT_MAP = {
    "anxiety":   ["nervousness", "fear"],
    "anger":     ["anger", "annoyance", "disapproval", "disgust"],
    "sadness":   ["sadness", "grief", "remorse", "disappointment"],
    "anhedonia": ["boredom", "neutral"],           # proxy
    "overwhelm": ["confusion", "nervousness"],     # partial proxy
    "positive_affect_present": ["joy", "amusement", "excitement",
                                "gratitude", "love", "optimism", "pride"],
}
```

Cross-mapped scores are binary (0/1). The LLM teacher model adds soft confidence scores (0.0–1.0) for ambiguous cases via Tier 3 consensus on top.

**PatternReframe/TherapistQA → `cognitive` adapter:**

These datasets have 10-label multi-label annotations. Direct 1-to-1 mapping; existing labels become binary confidence scores. Implemented in `build_sft_datasets.py`.

**SenticNet binary label → `stress_level` BN target:**

The existing 0/1 stress label (stressed / not-stressed) becomes the BN ground truth for parameter learning. **Never** pair BN evaluation texts with adapter SFT targets: keep SenticNet rows out of `sft_*` JSONL so adapter weights are not tuned on the same distribution that defines BN stress supervision.

#### Tier 3 — LLM Annotation with Multi-Run Consensus

For features requiring contextual inference — `behavioural_phenotypes`, `lexical` features, and `repetitive_thought_density` — annotate text from **Reddit Mental Health (Zenodo)** using a frontier model (GPT-4o):

Run each text through the teacher model (GPT-4o) **3–5 times** at temperature 0.35. Aggregate by averaging numeric scores; discard features where stdev > 0.25. The consensus protocol:

```python
def annotate_with_consensus(text, teacher_fn, n_runs=5, temp=0.35):
    all_outputs = [json.loads(teacher_fn(text, temperature=temp)) for _ in range(n_runs)]
    consensus = {}
    for feature in all_outputs[0]:
        scores = [o[feature] for o in all_outputs]
        mean, stdev = statistics.mean(scores), statistics.stdev(scores)
        consensus[feature] = mean if stdev < 0.25 else None  # flag ambiguous
    return consensus
```

Features where stdev exceeds ~0.25 are genuinely ambiguous — discard the example for that feature's training. Don't keep noisy labels.

#### Weak Signal Injection (All Tier 1B/3 Groups)

When generating frontier LLM annotation prompts from Reddit Mental Health (Zenodo) data, each prompt includes a `weak_signals` field containing related LIWC/TF-IDF/sentiment features already present in the source CSV files. These provide the frontier model with noisy-but-grounded context to improve annotation quality:

| Annotation group | Weak signals included in prompt |
|---|---|
| **Lexical** | `liwc_discrepancy`, `liwc_negations`, `liwc_negative_emotion`, `liwc_certainty`, `sent_neg`, `liwc_tentative`, `sent_compound` |
| **Affect** | `liwc_anxiety`, `liwc_positive_emotion`, `liwc_sadness`, `sent_neg`, `tfidf_stress`, `sent_pos` |
| **Behavioural** | `liwc_work`, `liwc_body`, `liwc_inhibition`, `liwc_health`, `tfidf_pain`, `tfidf_school`, `tfidf_therapist`, `tfidf_job`, `tfidf_sleep`, `tfidf_help`, `tfidf_therapi`, `tfidf_tire`, `tfidf_work`, `tfidf_support`, `domestic_stress_total`, `isolation_total` |
| **Repetitive thought** | `liwc_cognitive`, `liwc_insight` |

**Critical design principle:** Weak signals are provided **only to the frontier model during annotation**. They are **never included in the SFT datasets** — the LoRA-adapted student model learns to extract features from raw text alone, ensuring it doesn't depend on LIWC or TF-IDF features at inference time.

The prompt files are stored at `data/processed/llm_annotation_{group}.jsonl` and the frontier model responses at `data/processed/llm_annotation_{group}_responses*.jsonl`.

#### Affect Weak Signal Derivation

For the affect adapter's Reddit MH examples, the frontier LLM was asked to annotate **only** `anhedonia` and `overwhelm` — the two affect dimensions with no proxy dataset. The remaining four affect features (`anxiety`, `anger`, `sadness`, `positive_affect_present`) were derived from LIWC weak signals via calibrated thresholds in `build_sft_datasets.py::_affect_from_weak_signals()`:

- `anxiety_score = clamp01(liwc_anxiety / 0.016)` (normalised by p95)
- `sadness_score = clamp01(liwc_sadness / 0.024)` (normalised by p95)
- `anger_score = clamp01(sent_neg / 0.30 * 0.5)` (half-weight proxy)
- `positive_affect_present = clamp01((liwc_positive_emotion - 0.023) / 0.04)` (above-median)

Negative signal suppression: `positive_affect_present` is downweighted when strong negative signals dominate. These derived scores are combined with the LLM-annotated `anhedonia` and `overwhelm` to produce the final 6-feature affect training examples. The SFT dataset contains only the 6 feature scores — no LIWC values.

**Critical warning on cognitive distortions:** Prior works report low weighted F1 (0.2–0.4) for single-run LLM annotation, with considerable confusion between similar categories (catastrophizing vs. fortune-telling, overgeneralization vs. all-or-nothing). Multi-run consensus helps, but also consider **collapsing near-synonymous categories** where human review shows they're not distinguishable in your data (e.g., merge `fortune_telling` + `catastrophizing` → `negative_future_prediction` if κ < 0.4). Better to have 8 reliable features than 10 noisy ones going into the BN.

### 3.3 Per-Group Dataset Summary

| Adapter group | Tier | Primary data source | Annotation method | SFT examples |
|---|---|---|---|---|
| `lexical` | 1B | Reddit Mental Health (Zenodo) | Frontier LLM annotation (with LIWC weak signals in prompt) | ~5,000 |
| `cognitive` | 2 | PatternReframe + TherapistQA + ThinkingTrap | Cross-map from existing CD labels | ~10,000 |
| `affect` | 2+3 | GoEmotions (cross-map, all 6 features) + Reddit MH (LLM anhedonia/overwhelm + LIWC-derived other 4) | Cross-map + LLM + weak signal derivation | ~57,000 |
| `behavioural` | 3 | Reddit Mental Health (Zenodo) | Frontier LLM annotation (with LIWC/TF-IDF weak signals in prompt) | ~3,500 |
| `repetitive_thought` | 3 | Reddit Mental Health (Zenodo) | Frontier LLM annotation (with `liwc_cognitive`, `liwc_insight` in prompt) | ~2,500 |

### 3.4 Per-Group Output Schemas

Each adapter outputs a JSON with only its group's features:

**Lexical adapter output:**
```json
{"absolutist_word_density": 0.0, "negative_emotion_density": 0.0, "discrepancy_language": 0.0, "hedging_tentativeness": 0.0, "negation_aware_negative_sentiment": 0.0}
```

**Cognitive adapter output:**
```json
{"catastrophizing": 0.0, "all_or_nothing": 0.0, "overgeneralization": 0.0, "mind_reading": 0.0, "fortune_telling": 0.0, "emotional_reasoning": 0.0, "should_statements": 0.0, "mental_filter": 0.0, "magnification": 0.0, "disqualifying_positive": 0.0}
```

**Affect adapter output:**
```json
{"anxiety": 0.0, "anger": 0.0, "sadness": 0.0, "anhedonia": 0.0, "overwhelm": 0.0, "positive_affect_present": 0.0}
```

**Behavioural adapter output:**
```json
{"sleep_disruption": 0.0, "social_withdrawal": 0.0, "physical_exhaustion": 0.0, "work_overload": 0.0, "functional_impairment": 0.0, "help_seeking": 0.0}
```

**Repetitive thought adapter output:**
```json
{"repetitive_thought_density": 0.0}
```

**Tier 1A compute output:**
```json
{"first_person_singular_density": 0.0}
```

All values range `[0.0, 1.0]`.

### 3.5 Deliverables

- `data/raw/` — downloaded datasets (SenticNet Reddit_Combi, GoEmotions via HuggingFace `datasets`, PatternReframe, TherapistQA, Thinking Trap, [Reddit Mental Health — Zenodo](https://zenodo.org/records/3941387))
- `data/processed/llm_annotation_{group}.jsonl` — frontier model annotation prompts with `weak_signals` field (lexical, affect, behavioural, repetitive_thought)
- `data/processed/llm_annotation_{group}_responses*.jsonl` — frontier model annotation responses
- `data/sft/sft_{group}_train.jsonl` — per-group SFT training split for lexical, cognitive, affect, behavioural, repetitive_thought (**no BN eval texts, no LIWC/weak signals**)
- `data/sft/sft_{group}_val.jsonl` — per-group SFT validation split (**no BN eval texts, no LIWC/weak signals**)
- `output/bayesian_network*/bn_ground_truth_*.csv` — adapter feature vectors + stress labels for BN parameter learning and evaluation (generated during BN training)

---

## 4. Phase 2 — Per-Group Supervised Fine-Tuning with MS-Swift

### 4.1 Model Selection

**Target hardware: NVIDIA DGX Spark** (128 GB unified LPDDR5x, Grace Blackwell Superchip).

**Current base model:** `Qwen/Qwen2.5-7B-Instruct` with **LoRA** (full-precision base + LoRA adapters). Chosen for faster iteration. The base model weights are loaded once; each adapter is ~100–200 MB and can be swapped in/out without reloading the base.

**Scaling path:** `Qwen/Qwen2.5-32B-Instruct` with full LoRA, or `Qwen/Qwen2.5-72B-Instruct` with QLoRA (4-bit quantized base). The architecture supports scaling to larger models by changing `--model` and `--train_type` flags.

### 4.2 Training Configuration

Each of the **5** adapter groups (lexical, cognitive, affect, behavioural, repetitive_thought) is trained independently with the same hyperparameters via `train_adapters.py`:

```bash
# Train all 5 adapters sequentially:
python train_adapters.py

# Train specific adapter(s):
python train_adapters.py --groups affect cognitive

# Dry run — print commands without executing:
python train_adapters.py --dry_run
```

`train_adapters.py` builds and executes `swift sft` CLI commands for each group. Default configuration (7B + LoRA):

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | `all-linear` |
| Precision | bfloat16 |
| Epochs | 3 |
| Batch size | 4 × 4 gradient accumulation (effective 16) |
| Learning rate | 5e-5 (cosine schedule, 5% warmup) |
| Max sequence length | 2048 |
| Save strategy | Per epoch, keep last 2 |
| SFT data | `data/sft/sft_{group}_train.jsonl` / `val` |
| Output | `output/adapter_{group}/` |

Key choices:
- Each adapter trains on its own focused dataset (smaller, cleaner, tier-appropriate)
- Same LoRA rank/alpha across groups for consistency; tune per-group if evaluation shows need
- Cosine LR with warmup prevents overfitting on smaller per-group datasets
- Gradient checkpointing enabled to reduce memory usage
- NEFTune noise and DeepSpeed available as optional flags

### 4.3 Multi-Adapter Inference

At inference time, the base model is loaded **once**. Each adapter is run via `swift infer` CLI subprocess (one per group), matching the CLI pattern used for training. `first_person_singular_density` is computed via regex (no adapter needed).

The BN training scripts (`train_bayesian_senticnet.py`) handle this automatically: they locate the best checkpoint per adapter group, prepare input JSONL, run `swift infer`, and parse the JSON output.

```python
# Example: all 5 adapters + regex FPD in one pipeline call
result = {
    "first_person_singular_density": 0.08,   # regex
    "repetitive_thought_density": 0.42,       # adapter
    "absolutist_word_density": 0.72,           # adapter
    "catastrophizing": 0.88,                   # adapter
    "anxiety": 0.65,                           # adapter
    "sleep_disruption": 0.91,                  # adapter
    # ... all 29 features
}
```

### 4.4 Output Validation

Each adapter's output is validated against its group's feature schema. If output fails to parse:
1. Retry once with temperature 0.1
2. Fall back to uniform virtual evidence for that group's BN nodes (non-informative)
3. Log parse failure rate per adapter as a health metric

### 4.5 Deliverables

- `output/adapter_lexical/` — Lexical adapter LoRA weights
- `output/adapter_cognitive/` — Cognitive distortion adapter
- `output/adapter_affect/` — Affect adapter
- `output/adapter_behavioural/` — Behavioural phenotypes adapter
- `output/adapter_repetitive_thought/` — Repetitive thought adapter
- `train_adapters.py` — per-group or batch training script (builds `swift sft` CLI commands)

---

## 5. Phase 3 — Bayesian Network Construction (pgmpy)

### 5.1 Why pgmpy

| Criterion | pgmpy | pyAgrum | PyMC |
|---|---|---|---|
| Discrete BN with CPTs | Native | Native | Manual |
| **Virtual / soft evidence** | `TabularCPD` via `virtual_evidence` param | Supported | N/A |
| Structure learning | Hill-climb, PC, MMHC | GES, PC | N/A |
| Exact inference (Variable Elimination, Junction Tree) | Yes | Yes | MCMC only |
| Python ecosystem integration | Mature, pip-installable, active maintenance | Good | Different paradigm |

**pgmpy** is the best fit because it natively supports virtual evidence injection (critical for propagating LLM uncertainty), provides structure learning algorithms (HillClimbSearch with BIC scoring), and offers both exact and approximate inference.

### 5.2 Approach: Data-Driven Structure Learning

Rather than hand-designing a complex expert DAG with latent intermediate states (which produced near-random results in initial experiments), the final pipeline uses **data-driven structure learning** to discover the optimal BN topology from the adapter-extracted features.

**Pipeline:** `train_bayesian_senticnet.py`

1. **Prepare dataset:** Merge SenticNet Reddit_Combi (stressed) with EmpatheticDialogues (not-stressed) supplement
2. **Extract features:** Run all 5 LoRA adapters via `swift infer` + regex FPD on the merged dataset
3. **Discretize:** Adaptive thresholds (median per feature on training data) to convert continuous [0,1] scores to binary
4. **Structure learning:** `HillClimbSearch` with `BIC` scoring and expert constraints:
   - Forbid feature→feature edges (features are conditionally independent given stress)
   - Forbid feature→`stress_level` edges (stress causes features, not vice versa)
5. **Parameter learning:** `BayesianEstimator` with BDeu prior
6. **Inference:** Virtual (soft) evidence injection for each adapter output
7. **Classification threshold:** Optimised on training set (P(stressed) threshold = 0.83)

### 5.3 Learned Network Topology

Structure learning selected **11 of 29 features** — pruning 18 as redundant or non-informative. The resulting structure is naïve Bayes-like (all edges from `stress_level` to features):

```
                    stress_level (binary: not-stressed / stressed)
                         |
       ┌─────┬─────┬────┬────┬────┬────┬────┬────┬────┬────┐
       │     │     │    │    │    │    │    │    │    │
       ▼     ▼     ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
     func  negat neg_  anhe anger over  pos   rep   disc  abso  fpd
     _imp  _sent emo   doni       whelm _aff  _thou _lang _word
```

**Selected features (11):**

| # | Feature | Adapter Group |
|---|---|---|
| 1 | `functional_impairment` | Behavioural |
| 2 | `negation_aware_negative_sentiment` | Lexical |
| 3 | `negative_emotion_density` | Lexical |
| 4 | `anhedonia_affect` | Affect |
| 5 | `anger_affect` | Affect |
| 6 | `overwhelm_affect` | Affect |
| 7 | `positive_affect_present` | Affect |
| 8 | `repetitive_thought_density` | Repetitive Thought |
| 9 | `discrepancy_language` | Lexical |
| 10 | `absolutist_word_density` | Lexical |
| 11 | `first_person_singular_density` | Regex (Tier 1A) |

**Pruned groups:** All 10 cognitive distortions were dropped by BIC scoring (too subtle/noisy for binary stress detection). Most behavioural features were pruned (only `functional_impairment` survived). This aligns with the finding that lexical and affect features dominate binary stress detection.

### 5.4 Implementation

```python
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# 1. Structure learning with expert constraints
searcher = HillClimbSearch(train_data)
learned_model = searcher.estimate(
    scoring_method=BicScore(train_data),
    # Forbid feature→feature and feature→stress_level edges
)

# 2. Parameter learning with BDeu prior
model.fit(train_data, estimator=BayesianEstimator,
          prior_type="BDeu", equivalent_sample_size=10)

# 3. Inference: inject adapter outputs as virtual (soft) evidence
infer = VariableElimination(model)

adapter_outputs = {
    "functional_impairment": 0.91,
    "negative_emotion_density": 0.80,
    "anhedonia_affect": 0.72,
    "first_person_singular_density": 0.12,  # from regex
    # ... all 29 features (unused ones are ignored by the learned structure)
}

virtual_evidence = []
for node, prob_present in adapter_outputs.items():
    if node in model.nodes():
        ve = TabularCPD(
            variable=node, variable_card=2,
            values=[[1 - prob_present], [prob_present]],
        )
        virtual_evidence.append(ve)

result = infer.query(
    variables=["stress_level"],
    virtual_evidence=virtual_evidence,
)
# result.values = [P(not_stressed), P(stressed)]
```

### 5.5 Parameter Learning

CPT values are learned from the SenticNet dataset via `BayesianEstimator` with a BDeu prior (`equivalent_sample_size=10`). The classification threshold on `P(stressed)` is optimised on the training split (found to be 0.83).

Adaptive discretization: continuous adapter scores are binarised using per-feature median thresholds computed on the training set, rather than a fixed 0.5 cutoff. This handles features with skewed distributions (e.g., most posts have low `work_overload`).

### 5.6 Deliverables

- `train_bayesian_senticnet.py` — full pipeline: dataset preparation, adapter inference, structure learning, BN training, evaluation
- `output/bayesian_network_senticnet/` — saved BN model (`bn_model.pkl`), feature CSVs, evaluation results
---

## 6. Phase 4 — End-to-End Integration

### 6.1 Multi-Adapter Pipeline

```python
import json
from train_bayesian_senticnet import compute_tier1a
# adapter groups defined in train_bayesian_senticnet.py

def predict_stress(text: str, engine, bn_infer) -> dict:
    all_features = {}

    # Step 1: Tier 1A — pure compute
    all_features.update(compute_tier1a_features(text))

    # Step 2: Swap each adapter, extract group features
    for group, cfg in ADAPTER_GROUPS.items():
        adapter_path = find_best_checkpoint(cfg["output_dir"])
        raw = infer_with_adapter(engine, adapter_path, cfg["system_prompt"], text)
        group_features = parse_and_validate(raw, cfg["features"])
        if group_features is not None:
            all_features.update(group_features)
        else:
            # Fallback: uniform evidence (non-informative)
            for k in cfg["features"]:
                all_features[k] = 0.5

    # Step 3: Inject all 29 features into BN as soft evidence
    virtual_evidence = build_virtual_evidence(all_features)
    result = bn_infer.query(
        variables=["stress_level"],
        virtual_evidence=virtual_evidence,
    )

    return {
        "features": all_features,
        "stress_posterior": {
            "Low": float(result.values[0]),
            "Moderate": float(result.values[1]),
            "High": float(result.values[2]),
        },
    }
```

### 6.2 Fallback Handling

- If an adapter produces invalid JSON: retry once with temperature 0.1, then inject uniform virtual evidence for that group's nodes (non-informative — the BN marginalises it out gracefully).
- If a feature key is missing from the output: inject uniform evidence for that node only.
- If all adapters fail: return the prior distribution with a low-confidence flag.
- If Tier 1A compute returns 0 tokens (empty text): skip or flag.

### 6.3 Deliverables

- `train_bayesian_senticnet.py` — end-to-end pipeline: adapter inference + BN training + evaluation
- Fallback logic and JSON validation are embedded in the BN training scripts

---

## 7. Phase 5 — Evaluation

### 7.1 Component A: Per-Adapter Evaluation

Each adapter is evaluated independently against its group's gold-standard validation set:

| Metric | Method |
|---|---|
| JSON parse success rate | % of outputs that are valid JSON matching the group schema |
| Per-feature F1 | Compare extracted features against gold-standard (binarised at 0.5) |
| Macro-averaged F1 | Average F1 across features in the group |
| Calibration | Reliability diagram — do 80% confidence outputs match 80% ground truth rate? |
| Inter-adapter consistency | For features influenced by multiple groups, check agreement |

### 7.2 Component B: Bayesian Network

| Metric | Method |
|---|---|
| Stress classification accuracy | Compare MAP estimate of `stress_level` to SenticNet ground truth |
| Macro F1 (3-class) | Across Low / Moderate / High stress |
| AUC-ROC | One-vs-rest for each stress level |
| Sensitivity analysis | Vary individual CPT entries ±10% and measure posterior stability |
| Ablation | Remove each adapter group entirely and measure accuracy drop |

### 7.3 End-to-End

| Metric | Method |
|---|---|
| End-to-end stress detection F1 | Full multi-adapter pipeline on held-out SenticNet test set |
| Adapter ablation | Drop each adapter group (use uniform evidence for its nodes) and measure F1 change |
| Tier 1A vs LLM | Compare pure-compute `first_person_singular_density` against an LLM-extracted version |
| Explainability | For each prediction, output the top-3 evidence nodes that most shifted the posterior |
| Latency | Time from raw text to final posterior; target < 10s on GPU (5 sequential adapter calls + Tier 1A regex compute) |

---

## 8. Timeline

| Week | Tasks | Owner |
|---|---|---|
| **7–8** | Literature review finalised. All datasets downloaded and explored. BN topology designed. Per-group annotation rubrics written and validated. Tier 2 cross-mapping implemented. | All |
| **9** | Per-group SFT datasets constructed. Initial LoRA fine-tune for each adapter. Validate per-group JSON output quality. | Benjamin |
| **9** | BN implemented in pgmpy. CPTs set from literature. Inference tested with synthetic inputs. | Gerald |
| **9** | Evaluation framework designed. Gold-standard validation set prepared (Tier 4). | Van Phuc |
| **10** | Per-adapter iteration (self-training expansion, hyperparameter tuning per group). CD category collapse analysis. | Benjamin |
| **10** | CPT parameter learning from per-group annotated data. Sensitivity analysis on BN. | Gerald |
| **10** | Belief calibration analysis per adapter. Multi-adapter integration pipeline. | Van Phuc |
| **11–12** | End-to-end integration and testing. Adapter ablation studies. Explainability module. | All |
| **13** | Report writing, presentation preparation, final evaluation runs. | All |

---

## 9. Technical Requirements

### 9.1 Dependencies

```
ms-swift>=3.0
torch>=2.1
pgmpy>=1.0
transformers>=4.40
datasets
pandas
numpy
scikit-learn
matplotlib
spacy>=3.0
```

### 9.2 Hardware

- **SFT Training (DGX Spark):** 128 GB unified memory (Grace Blackwell). Train 5 adapters sequentially on the same base model. ~100–200 MB per adapter.
- **Inference:** Same node. Base model loaded once; adapters swapped in/out (minimal additional memory per adapter).
- **Bayesian Network:** CPU-only, runs in seconds.

### 9.3 Repository Structure

```
3263-assignment/
├── build_sft_datasets.py          # Per-group SFT JSONL construction
├── train_adapters.py              # LoRA training via MS-Swift (all 5 groups)
├── train_bayesian_senticnet.py    # SenticNet BN pipeline (best model)
├── train_bayesian_network.py      # Dreaddit BN pipeline (experimental)
├── train_bayesian_deptweet.py     # DepTweet BN pipeline (experimental)
├── implementation_plan.md         # This file
├── report.md                      # Technical report
├── dataset_columns.md             # Dataset column reference
├── data/
│   ├── raw/                       # Original dataset downloads
│   │   ├── reddit_mental_health/  # 108 CSVs (~3.14 GB) — SFT text source
│   │   ├── senticnet_stress/      # Reddit_Combi — BN primary dataset
│   │   ├── patternreframe/        # Cognitive cross-map source
│   │   ├── therapistqa_cd/        # Cognitive cross-map source
│   │   ├── thinkingtrap/          # Cognitive cross-map source
│   │   └── depression_severity/   # DepTweet BN variant
│   ├── processed/                 # LLM annotation prompts + responses
│   │   ├── llm_annotation_{group}.jsonl           # Prompts (with weak_signals)
│   │   └── llm_annotation_{group}_responses.jsonl # Frontier model responses
│   └── sft/                       # Per-group SFT train/val JSONL
│       ├── sft_{group}_train.jsonl    # No LIWC, no BN eval texts
│       └── sft_{group}_val.jsonl
├── output/
│   ├── adapter_lexical/           # LoRA weights + training logs
│   ├── adapter_cognitive/
│   ├── adapter_affect/
│   ├── adapter_behavioural/
│   ├── adapter_repetitive_thought/
│   ├── bayesian_network_senticnet/  # Best BN model + results
│   ├── bayesian_network/            # Dreaddit BN (experimental)
│   └── bayesian_network_deptweet/   # DepTweet BN (experimental)
└── venv/                          # Python virtual environment
```
