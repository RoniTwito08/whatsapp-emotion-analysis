# Hebrew Sales Dataset Generator

A production-ready Python pipeline that generates synthetic Hebrew WhatsApp business conversations for research purposes using the OpenAI API.

## What This Project Does

The generator extends an approved seed dataset (`data/corpus_300.json`) to a target of 5,000 conversations. It:

1. Loads the approved 300-conversation source dataset.
2. Creates a diverse, structured conversation *plan* in Python before calling the LLM.
3. Sends one plan at a time to the LLM and receives only the message texts.
4. Calculates all deterministic fields in Python (timestamps, delays, interest scores, behavioral features).
5. Detects and rejects duplicates at the message and conversation level.
6. Validates schema coherence against the exact structure of `data/corpus_300.json`.
7. Saves progress after every accepted conversation for safe resume.
8. Writes generation and validation reports.

## Source of Truth

`data/corpus_300.json` is the single source of truth for schema, field names, allowed values, and behavioral conventions. **Do not modify this file.**

## Project Structure

```
hebrew-sales-dataset-generator/
├── .env.example            ← Copy to .env and fill in your API key
├── requirements.txt
├── generate.py             ← Main CLI
├── validate_dataset.py     ← Validation CLI
├── data/
│   └── corpus_300.json     ← Source dataset (read-only)
├── output/
│   ├── generated_conversations.json   ← Only new conversations
│   ├── combined_dataset.json          ← Source + generated
│   └── checkpoint.json               ← Resume state
├── reports/
│   ├── generation_report.json
│   ├── validation_report.json
│   └── rejected_conversations.jsonl
├── src/
│   ├── config.py
│   ├── models.py
│   ├── dataset_loader.py
│   ├── scenario_planner.py     ← Creates conversation plans
│   ├── prompt_builder.py       ← Builds LLM prompts
│   ├── llm_client.py           ← OpenAI client with Structured Outputs
│   ├── conversation_builder.py ← Assembles full conversation object
│   ├── feature_calculator.py   ← Deterministic behavioral features
│   ├── similarity_checker.py   ← Duplicate / similarity detection
│   ├── validator.py            ← Schema + coherence validation
│   ├── checkpoint_manager.py   ← Safe save/restore
│   └── generator.py            ← Main orchestration loop
└── tests/
    ├── test_dataset_loader.py
    ├── test_feature_calculator.py
    ├── test_similarity_checker.py
    ├── test_validator.py
    ├── test_checkpoint_manager.py
    └── test_generator.py
```

## Setup

### 1. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Available models: `gpt-4o-mini` (cheaper), `gpt-4o` (higher quality), `gpt-5-mini`.

## Usage

### Dry-run (no API calls)

```bash
python generate.py --count 5 --dry-run
```

Prints conversation plans without making API calls. Safe to run without a key.

### Generate 20 conversations (initial test)

```bash
python generate.py --count 20
```

This is the recommended first run. Output goes to `output/combined_dataset.json`.

### Reproducible generation

```bash
python generate.py --count 20 --seed 42
```

### Validate the output

```bash
python validate_dataset.py output/combined_dataset.json
```

### Resume an interrupted run

```bash
python generate.py --target-total 5000 --resume
```

### Generate until 5,000 total

```bash
python generate.py --target-total 5000
```

**Warning:** This will make ~4,700 API calls. Monitor costs before running.

### Windows PowerShell equivalents

```powershell
python generate.py --count 20 --dry-run
python generate.py --count 20
python validate_dataset.py output\combined_dataset.json
python generate.py --target-total 5000 --resume
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use a `MockLLMClient` and never call the real API.

## Where Reports Are Written

| File | Contents |
|------|----------|
| `output/generated_conversations.json` | Only newly generated conversations |
| `output/combined_dataset.json` | Source (300) + generated conversations |
| `output/checkpoint.json` | Resume state (accepted count, ID, distributions) |
| `reports/generation_report.json` | Statistics, token usage, domain distribution |
| `reports/validation_report.json` | Schema validation results |
| `reports/rejected_conversations.jsonl` | Every rejected generation with reason |

## How Duplicate Detection Works

The system uses four layers:

1. **Exact message match** — normalized text is indexed in a hash map. Any exact match is rejected immediately.
2. **Fuzzy message similarity** — RapidFuzz `token_sort_ratio` against candidate messages retrieved from length and prefix buckets. Threshold: 88 (configurable via `MESSAGE_SIMILARITY_THRESHOLD`).
3. **Conversation-level TF-IDF cosine similarity** — character n-gram TF-IDF vectors. Threshold: 82 (configurable via `CONVERSATION_SIMILARITY_THRESHOLD`).
4. **Structural signature tracking** — overused `domain|trajectory|outcome|length|roles` combinations are deprioritized.

A forbidden phrase list (sourced from the rejected `corpus_new_301_500.json`) is also injected into every LLM prompt.

## API Cost and Rate Limit Considerations

- Each conversation generation makes one API call (one conversation per call).
- Generating 4,700 conversations with `gpt-4o-mini` costs approximately **$5–15** depending on conversation length.
- With `gpt-4o`, costs are approximately 10× higher.
- The generator uses exponential backoff for rate limit errors (429) and temporary server errors (5xx).
- Set `GENERATION_BATCH_SIZE` to control how many generations to run before pausing if needed.
- Use `--resume` freely — progress is saved after every accepted conversation.

## Generation Architecture

```
Load existing dataset
        ↓
Generate a unique conversation plan (Python)
        ↓
Send one plan to the LLM → receive message texts only
        ↓
Build the complete conversation object (Python)
        ↓
Calculate all deterministic fields (Python)
        ↓
Validate schema and coherence
        ↓
Check duplicates and similarity
        ↓
Accept or reject → log rejection
        ↓
Save checkpoint
```

The LLM is responsible **only** for writing the Hebrew message texts. Python calculates all IDs, timestamps, delays, scores, and behavioral features.

## Limitations and Risks

- **Model bias:** The LLM may over-represent certain conversation structures, customer personas, or phrase patterns despite anti-repetition controls. Human review is recommended before research use.
- **Hebrew quality:** LLM output may occasionally use non-natural phrasing, over-formal register, or incorrect grammatical gender.
- **Slash-gender forms:** The generator explicitly rejects messages containing forms like `מתעניין/ת`. Note that the source corpus (`corpus_300.json`) contains some such forms from an earlier generation pass.
- **Similarity thresholds:** The default thresholds (88 / 82) may be too strict or too loose depending on the research goal. Tune via environment variables.
- **Human review:** Synthetic data should be reviewed by a native Hebrew speaker before use in research, training, or evaluation.
