# text-model

AlephBERT fine-tuning pipeline for binary Hebrew conversation classification.
Classifies WhatsApp sales conversations as `interested` or `losing_interest`
based on customer message text only.

---

## What this model does

Fine-tunes `onlplab/alephbert-base` (Hebrew BERT) for sequence classification
on conversation-level data. The classifier reads the customer's messages from
a conversation and predicts whether the customer is `interested` (likely to
convert) or `losing_interest` (ghosting, rejecting, churning).

**This model has not been evaluated yet.** Metrics will be reported here once
a full training run is completed. Do not interpret code structure or loss values
from smoke tests as model performance.

---

## Dataset

**Source:** `hebrew-sales-dataset-generator/output/combined_dataset.json`

- 3,055 synthetic Hebrew WhatsApp-style conversations
- All conversations are labeled `synthetic: true`
- Binary label distribution: `interested` = 1,529 (50.0%), `losing_interest` = 1,526 (50.0%)
- 58 business domains (furniture, dental, legal, fitness, etc.)

### Label mapping

Raw `final_outcome` → binary label:

| Binary label | Raw outcomes |
|---|---|
| `interested` | `converted`, `appointment_set`, `pending`, `reengaged_pending` |
| `losing_interest` | `explicit_rejection`, `competitor_loss`, `delivery_loss`, `trust_loss`, `ghosted` |

Label IDs are assigned alphabetically: `interested = 0`, `losing_interest = 1`.

### Model input

Only **customer messages** are fed to the tokenizer. Business messages,
`interest_label`, `interest_score`, `interest_trajectory`, and all outcome
fields are excluded. The only text that enters the model is
`msg["text"]` where `msg["role"] == "customer"`.

Customer messages within a conversation are joined with ` [SEP] `.

---

## Train / validation / test split

- **70 / 15 / 15** split at the **conversation level** (stratified by binary label)
- Random seed: 42
- Train: 2,137 conversations | Val: 459 | Test: 459
- Split is computed once and saved to `outputs/split_ids.json` so the
  TF-IDF baseline can use the identical test set for a fair comparison

**No conversation ID appears in more than one split.** The code explicitly
validates zero overlap after splitting.

---

## Truncation warning

Hebrew text is tokenized into subword tokens by AlephBERT's tokenizer.
Estimated truncation at different `max_length` values:

| max_length | Estimated % truncated |
|---|---|
| 128 | ~94% |
| 256 | ~85% |
| **512 (current default)** | **~38%** |

Using `max_length=256` is **not recommended** — it would truncate 85% of
conversations and cause the model to miss most of the conversational context.
The current default is **512**, which is AlephBERT's architectural maximum.

Even at 512, ~38% of conversations are truncated. A future experiment
should test full-conversation summarization or hierarchical encoding.

---

## Leakage risks

1. **Per-message `interest_label`**: Every message has a label field
   (e.g., `"converted"`, `"rejected"`). The current pipeline reads only
   `msg["text"]` — this field never enters the model. **Safe as-is.**

2. **Terminal customer messages**: Converted conversations end with
   commitment language; rejected ones with rejection language. This is
   a natural correlation (not a bug), but in synthetic data these patterns
   may be more stereotyped than in real data, which would inflate reported
   performance. Results should be validated on real conversation data before
   drawing conclusions.

3. **Fields that must never enter the model:**
   `final_outcome`, `interest_trajectory`, `final_interest_score`,
   `interest_score` per message.

---

## Files

```
text-model/
├── config.json                  # All configurable parameters
├── train.py                     # Training pipeline (entry point)
├── analyze_dataset.py           # Dataset analysis report
├── splitter.py                  # Stratified conversation-level split
├── dataset.py                   # HebrewConversationDataset
├── data_loader.py               # Config/tokenizer/model/dataloader factories
├── validate_inputs.py           # Input pipeline smoke test
├── inspect_batch.py             # Manual batch inspection tool
├── behavioral_features_design.py # Design sketch for behavioral features
├── requirements.txt
├── tests/
│   ├── test_splitter.py
│   ├── test_label_mapping.py
│   └── test_train_smoke.py
├── data/
│   └── sample_corpus.jsonl      # Tiny test fixture only — NOT for training
└── outputs/
    └── alephbert_baseline_v1/   # Created on first training run
        ├── best_model/          # Saved checkpoint + tokenizer
        ├── config.json
        ├── split_statistics.json
        ├── training_history.json
        ├── test_metrics.json
        ├── classification_report.txt
        ├── confusion_matrix.csv
        └── test_predictions.csv
```

---

## Setup

```bash
cd text-model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first training run will download `onlplab/alephbert-base` (~500 MB) from
Hugging Face into `~/.cache/huggingface/`. Subsequent runs use the cache.

---

## Running training

```bash
# Basic — uses all defaults from config.json
python train.py --config config.json

# Override epochs
python train.py --config config.json --epochs 3

# Custom experiment name (saves to outputs/my_experiment/)
python train.py --config config.json --experiment-name alephbert_v2
```

Training prints per-epoch metrics to stdout and saves the best model
(by validation macro F1) as a checkpoint. After training, it loads the
best checkpoint and evaluates once on the held-out test set.

**Expected runtime** (rough estimates):
- CPU only: ~8–15 hours for 5 epochs on 3,055 conversations at max_length=512
- MPS (Apple Silicon): ~2–4 hours
- CUDA (GPU): ~30–60 minutes

---

## Running analysis

```bash
python analyze_dataset.py --config config.json
```

Reports: class distribution, raw outcome counts, conversation length
distributions, estimated truncation percentage at common max_length values,
domain distribution, leakage warnings.

---

## Running the input pipeline smoke test

```bash
python validate_inputs.py --config config.json
```

Verifies tensor shapes, dtypes, value ranges, and runs a single forward
pass. Does **not** train the model. Requires downloading AlephBERT.

---

## Configuration

All key parameters live in `config.json`:

| Section | Key | Default | Notes |
|---|---|---|---|
| `data.input_path` | — | `../hebrew-sales-dataset-generator/output/combined_dataset.json` | Real dataset |
| `data.included_roles` | — | `["customer"]` | Only customer text enters the model |
| `tokenizer.max_length` | — | `512` | AlephBERT maximum; do not use 256 |
| `split.train_ratio` | — | `0.70` | Conversation-level stratified split |
| `split.random_seed` | — | `42` | Fixed for reproducibility |
| `training.epochs` | — | `5` | Override with `--epochs` flag |
| `training.learning_rate` | — | `2e-5` | AdamW LR |
| `training.weight_decay` | — | `0.01` | AdamW weight decay |
| `training.warmup_ratio` | — | `0.1` | Linear warmup fraction of total steps |
| `training.gradient_clip` | — | `1.0` | Gradient clipping |
| `training.positive_class` | — | `losing_interest` | Positive class for binary F1 |
| `dataloader.train_batch_size` | — | `8` | Reduce to 4 if OOM |
| `dataloader.eval_batch_size` | — | `16` | For val/test |

---

## How the best checkpoint is selected

After each epoch, validation metrics are computed (loss, accuracy, precision,
recall, F1, macro F1, weighted F1). The epoch whose **validation macro F1**
is highest is saved as `best_model/`. Macro F1 is used (not accuracy) because
it equally weights both classes regardless of any residual class imbalance.

Configure `training.save_best_metric` in `config.json` to change the metric.

---

## Output files

After a successful run, `outputs/<experiment_name>/` contains:

| File | Contents |
|---|---|
| `best_model/` | Saved AlephBERT weights + tokenizer (load with `from_pretrained`) |
| `config.json` | Copy of the config used for this run |
| `split_statistics.json` | Train/val/test conversation counts |
| `training_history.json` | Per-epoch train loss and all val metrics |
| `test_metrics.json` | Final test set metrics (loss, acc, P/R/F1, macro F1) |
| `classification_report.txt` | Per-class sklearn classification report |
| `confusion_matrix.csv` | Labeled confusion matrix |
| `test_predictions.csv` | Per-conversation: ID, actual label, predicted label, probabilities |

`outputs/split_ids.json` (at the top of `outputs/`) contains train/val/test
IDs so the TF-IDF baseline can use the same test set.

---

## Running the TF-IDF baseline on the same split

After the AlephBERT pipeline has run once (producing `outputs/split_ids.json`):

```bash
cd ../nlp-baseline
python train_baseline.py \
  --input ../hebrew-sales-dataset-generator/output/combined_dataset.json \
  --config config_messages.json \
  --output results/combined_shared_split \
  --split-ids ../text-model/outputs/split_ids.json
```

Without `--split-ids`, the baseline uses its own independent 80/20 split
(original behavior, not directly comparable to AlephBERT).

---

## Running on CPU / GPU / MPS

The pipeline automatically selects the best available device in order:
`cuda > mps (Apple Silicon) > cpu`.

To force CPU, set `CUDA_VISIBLE_DEVICES=""` before running:
```bash
CUDA_VISIBLE_DEVICES="" python train.py --config config.json
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

Tests cover: stratified split correctness, zero ID overlap, label mapping,
split ID serialization/deserialization, dataset `subset_ids` filtering,
metrics correctness, and checkpoint save/load. All tests run without
downloading any model.

---

## Known limitations of the synthetic dataset

1. **All synthetic.** Every conversation was generated by an LLM. Performance
   on real Israeli business WhatsApp conversations may be lower.

2. **Stereotyped terminal messages.** Synthetic converted conversations end
   with commitment-language phrases; rejected conversations end with
   rejection-language phrases. A model can exploit these patterns instead of
   learning conversational trajectory, inflating accuracy.

3. **38% truncation at max_length=512.** Long conversations lose their
   later messages. The most informative messages (commitment/rejection
   signals) often appear later, so this is a meaningful limitation.

4. **No real-world validation.** No human-annotated held-out set exists yet.
   Test metrics should not be presented as production-quality estimates.
