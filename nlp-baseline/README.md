# nlp-baseline

A reusable baseline NLP pipeline for classifying WhatsApp-style conversations
or flat labeled text. It performs **text cleaning → TF-IDF feature extraction
→ Logistic Regression training → evaluation → saving all results**, driven
entirely by a JSON config file so the script itself never needs to change
when a new corpus arrives.

This is a **baseline**: its purpose is to give a fast, reproducible,
well-measured starting point to compare future models against — not to be
the final model.

## What it does

1. Loads a corpus in **JSONL**, **JSON**, or **CSV** format.
2. Reads all corpus-specific behavior from a config file:
   - which field holds the record id / label,
   - whether text comes from a single field or from a list of conversation
     messages (with role filtering and a join separator),
   - nested field paths, using dot notation (e.g. `"customer.contact.email"`),
   - text cleaning rules,
   - a label mapping (raw label → training class), with any unmapped label
     ignored,
   - TF-IDF and Logistic Regression hyper-parameters.
3. Cleans text: replaces URLs/emails/phone numbers with placeholder tokens,
   optionally replaces numbers, strips stray symbols, normalizes whitespace —
   all while **preserving Hebrew text** (Python's Unicode-aware string/regex
   handling never touches Hebrew letters).
4. Builds a scikit-learn `Pipeline([TfidfVectorizer, LogisticRegression])`.
5. Performs a **stratified** train/test split.
6. Trains, evaluates (accuracy, precision, recall, F1, macro F1, weighted F1,
   a full classification report, and a confusion matrix), and saves
   everything needed to inspect or reuse the model.
7. Works for both **binary** and **multiclass** labels.

## Project structure

```
nlp-baseline/
├── train_baseline.py     # main training/evaluation script
├── predict.py             # load a saved model and classify one piece of text
├── config_messages.json   # config for conversation-based corpora (format A)
├── config_csv.json        # config for flat text corpora (format B)
├── requirements.txt
├── README.md
├── data/                  # put your corpus file(s) here
│   └── .gitkeep
└── results/               # default output directory (created if missing)
    └── .gitkeep
```

## Setup (Windows)

From inside `nlp-baseline/`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Where to put the corpus

Place your corpus file inside `data/`, e.g. `data/corpus.jsonl` or
`data/corpus.csv`.

## Supported corpus shapes

**A. Conversation-based** (use `config_messages.json`) — one record per
conversation, with a list of `{role, text}` messages and a final outcome
label:

```json
{
  "conversation_id": "conv_001",
  "final_outcome": "converted",
  "messages": [
    {"role": "customer", "text": "כמה זה עולה?"},
    {"role": "business", "text": "המחיר הוא 500 ש״ח"}
  ]
}
```

**B. Flat text** (use `config_csv.json`) — one record per labeled text:

```json
{"id": "1", "text": "כמה זה עולה?", "label": "interested"}
```

Both shapes work in any of the three file formats (JSONL/JSON/CSV) — the
shape is determined by the config's `text_mode` (`"messages"` or `"field"`),
not by the file extension.

## Running the script

Conversation-based corpus (JSONL), using the messages config:

```powershell
python train_baseline.py --input data/corpus.jsonl --config config_messages.json --output results
```

Flat text corpus (CSV), using the CSV config:

```powershell
python train_baseline.py --input data/corpus.csv --config config_csv.json --output results
```

The same commands work with a `.json` input file instead of `.jsonl`/`.csv` —
the loader auto-detects format from the extension.

## Adapting the config when the final corpus arrives

Both config files are plain JSON; edit the copy that matches your corpus
shape (or make a new copy per corpus):

- `id_field` / `label_field` / `text_field` / `messages_field` /
  `role_field` / `message_text_field` — field names in your records. Use
  dot notation for nested fields, e.g. `"metadata.outcome"`.
- `included_roles` — which message roles to keep (e.g. `["customer"]`, or
  `["customer", "business"]` to include both).
- `message_separator` — string used to join included messages.
- `label_mapping` — maps each training class to the list of raw label
  values that belong to it. Any record whose raw label is **not** listed
  under any class is **ignored**. Remove this key entirely to use raw label
  values as-is (no mapping/filtering).
- `minimum_examples_per_class` — classes with fewer examples than this are
  dropped before training (with a warning), since a stratified split needs
  every remaining class adequately represented.
- `test_size`, `random_state` — train/test split behavior.
- `cleaning` — toggle URL/email/phone/number replacement, English
  lowercasing, and stray-symbol removal.
- `tfidf` — `analyzer`, `ngram_range`, `min_df`, `max_df`, `max_features`,
  `sublinear_tf`.
- `model` — `max_iter`, `class_weight`, `C`, `solver`, `random_state`.

No code changes are needed for a new corpus — only the config.

## Output files

Every run writes these files into the `--output` directory (created if it
doesn't exist):

| File | Contents |
|---|---|
| `tfidf_logistic_regression.joblib` | The fitted `Pipeline` (TF-IDF + Logistic Regression), loadable with `joblib.load` |
| `metrics.json` | Accuracy, precision, recall, F1, macro F1, weighted F1, train/test counts, class list |
| `classification_report.txt` | scikit-learn's per-class classification report |
| `confusion_matrix.csv` | Confusion matrix with labeled rows (`actual_*`) and columns (`pred_*`) |
| `test_predictions.csv` | Per test record: `record_id`, `cleaned_text`, `actual_label`, `predicted_label`, and one `probability_<class>` column per class |

## Running predict.py

```powershell
python predict.py --model results/tfidf_logistic_regression.joblib --text "כמה זה עולה?"
```

This prints the predicted class and the model's probability for every class
it was trained on. It applies the same default text-cleaning steps used at
training time (see `DEFAULT_CLEANING_CONFIG` at the top of `predict.py`) —
adjust that constant if your final config changes the cleaning rules.

## A note on small datasets

This repository currently contains only small/sample datasets. Metrics
computed on very few examples (especially per class) are **not reliable** —
a handful of test examples can swing accuracy or F1 by large margins, and a
stratified split can fail outright if a class has too few members. Treat
early results as a pipeline smoke test, not a real performance measurement.
Re-run once your teammate provides the final corpus, ideally with dozens of
examples per class or more.
