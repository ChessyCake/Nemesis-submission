# Redrob Hackathon Submission

This repository contains a top-100 candidate ranker for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## What it does

The ranker reads `candidates.jsonl` from the repository root, scores candidates against the released job description, filters honeypots, and writes a `submission.csv` file with the required columns:

`candidate_id,rank,score,reasoning`

## Setup

Use Python 3.11 or similar, then install the dependencies:

```bash
pip install -r requirements.txt
```

If your candidate pool is still compressed, you can keep `candidates.jsonl.gz` in the repo root instead of extracting it manually.

## Reproduce the submission CSV

Run the ranker from the repository root:

```bash
python ranker_final.py
```

That command will:

1. Load `candidates.jsonl`, or fall back to `candidates.jsonl.gz` if needed.
2. Filter honeypot candidates.
3. Rank the top 100 candidates.
4. Validate the top-100 IDs and honeypot rate.
5. Write `submission.csv` in the repository root.

## Files in this repo

- `ranker_final.py` - ranking pipeline and CSV export
- `submission_metdata.yaml` - portal metadata for the submission
- `submission.csv` - generated submission file

## Notes

- The ranking step is CPU-only and does not call hosted LLM APIs.
- The model uses sentence-transformers when available ( if the packages are installed correctly) and falls back to TF-IDF + SVD if not.
- The submission must contain exactly 100 rows, ranked from 1 to 100.
