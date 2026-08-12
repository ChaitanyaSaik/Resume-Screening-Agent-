# Resume Screening Agent

An AI agent that ranks a folder of resumes (PDF / DOCX / TXT) against a job
description using NLP similarity + explainable skill/experience matching,
and outputs a scored, ordered shortlist as CSV and JSON.

Built for the Rooman AI Challenge — 24-Hour AI Agent Challenge
(Category 1 — HR & Recruitment: Resume Screening Agent).

> **"My agent takes a job description + a folder of resumes, and produces a
> ranked, scored shortlist with a plain-English reason for every score."**

---

## What it does

1. Reads a job description (`.txt`, `.pdf`, or `.docx`).
2. Reads every resume in a folder (`.pdf`, `.docx`, `.txt` — mixed formats supported).
3. Extracts skills, years of experience, education level, email, and phone from each.
4. Scores every resume against the JD using:
   - TF-IDF + cosine similarity (textual relevance)
   - Skill overlap against the JD's required skills
   - Experience match against the JD's stated requirement
5. Ranks candidates highest → lowest score.
6. Writes `output/ranked_candidates.csv` and `output/ranked_candidates.json`.
7. Prints a readable summary (score, matched/missing skills, reasoning) to the console.

Works **completely offline / free** by default (no API key required — scoring
and reasoning are rule-based). An optional `--use-llm` flag can turn on
Claude-generated reasoning text if you have an Anthropic API key.

---

## Project structure

```
resume-screening-agent/
├── main.py                  # CLI entry point — run this
├── requirements.txt
├── SCORING_METHOD.md        # How the score is calculated (deliverable)
├── README.md                # This file
├── .env.example              # Template for optional API key
├── src/
│   ├── parser.py            # Reads PDF/DOCX/TXT into plain text
│   ├── extractor.py         # Extracts skills, experience, education, contact info
│   ├── scorer.py             # TF-IDF similarity + skill overlap + experience scoring
│   └── reasoner.py           # Generates the human-readable "why this score" text
├── tests/                    # pytest test suite (55 tests) — see "Running the test suite" below
├── conftest.py                # makes `src`/`main` importable from tests
├── data/
│   ├── job_description.txt  # Sample JD (Backend Python Developer)
│   └── resumes/              # 12 sample resumes — mixed .txt / .pdf / .docx
└── output/                   # Generated after running — CSV + JSON results
```

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd resume-screening-agent

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

That's it — no API key is required for the default run.

### Optional: enable LLM-generated reasoning

Only needed if you want to pass `--use-llm`.

```bash
cp .env.example .env
# then edit .env and add:
# ANTHROPIC_API_KEY=your-key-here
```

---

## How to run

Run it against the included sample data:

```bash
python main.py --jd data/job_description.txt --resumes data/resumes --out output
```

You'll see console output like:

```
Resume Screening Agent
=======================
Job description : data/job_description.txt
Resumes folder   : data/resumes
LLM reasoning    : OFF (rule-based)

[1/4] Loaded job description (1369 chars). Detected 17 required skill(s): [...]
[2/4] Parsed 12 resume(s).
[3/4] Scored and ranked 12 candidate(s).
[4/4] Generated reasoning for each candidate.

Outputs written:
  - output/ranked_candidates.csv
  - output/ranked_candidates.json

Top 10 Candidates
============================================================
1. resume_03_daniel_okafor.pdf  —  Score: 48.61/100
   Skills matched : agile, aws, ci/cd, docker, fastapi, flask, gcp, git, kubernetes, postgresql, python, redis
   Skills missing : communication, django, machine learning, mysql, nosql
   Experience     : 6 yr(s) (JD wants 3+)
   Education      : Master
   Reasoning      : Candidate matches 12 required skill(s) (...); is missing 5 required skill(s) (...); meets the 3+ year experience requirement with 6 year(s); holds a Master-level qualification.
------------------------------------------------------------
...
```

### Run it on your own data

```bash
python main.py --jd path/to/your_jd.txt --resumes path/to/your/resumes_folder --out output
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--jd` | `data/job_description.txt` | Path to the job description (`.txt`, `.pdf`, `.docx`) |
| `--resumes` | `data/resumes` | Folder containing resumes to screen |
| `--out` | `output` | Folder to write `ranked_candidates.csv` / `.json` |
| `--use-llm` | off | Use Claude for the reasoning text (needs `ANTHROPIC_API_KEY`) |
| `--top` | `10` | How many top candidates to print to the console |

---

## Running the test suite

The project includes a full `pytest` test suite (55 tests) covering every
module: file parsing (PDF/DOCX/TXT), field extraction (skills/email/phone/
experience/education), scoring (TF-IDF similarity, skill overlap,
experience match), reasoning (including the LLM-fallback safety path), and
an end-to-end integration test that runs the whole agent against the real
sample data in `data/`.

```bash
# pytest is already in requirements.txt, so it's installed by:
# pip install -r requirements.txt

python -m pytest -v
```

Expected result: `55 passed`.

```
tests/
├── test_parser.py       # PDF/DOCX/TXT reading, unsupported formats, empty files
├── test_extractor.py     # skills, email, phone, experience years, education
├── test_scorer.py        # TF-IDF similarity, skill overlap, experience match
├── test_reasoner.py      # rule-based reasoning text + LLM fallback safety
└── test_integration.py   # runs main.run() end-to-end on the real sample data
```

Run a single file if you only want to check one part, e.g.:

```bash
python -m pytest tests/test_scorer.py -v
```

---

## Sample input / output

- **Sample JD:** [`data/job_description.txt`](data/job_description.txt) — a Backend
  Python Developer role.
- **Sample resumes:** [`data/resumes/`](data/resumes) — 12 fictional resumes deliberately
  spanning strong fits (senior backend engineers), partial fits (frontend/data/ML
  engineers with some overlapping skills), and poor fits (marketing, product,
  no technical background), in a **mix of `.txt`, `.pdf`, and `.docx`** formats
  to demonstrate multi-format parsing.
- **Sample output:** [`output/ranked_candidates.csv`](output/ranked_candidates.csv) and
  [`output/ranked_candidates.json`](output/ranked_candidates.json) — generated by running
  the command above.

Example CSV row:

```csv
rank,filename,final_score,tfidf_score,skill_score,experience_score,experience_years,education_level,email,phone,matched_skills,missing_skills,reasoning
1,resume_03_daniel_okafor.pdf,48.61,11.95,70.59,100.0,6,Master,daniel.okafor@example.com,+234 802 555 0199,"agile, aws, ci/cd, docker, fastapi, flask, gcp, git, kubernetes, postgresql, python, redis","communication, django, machine learning, mysql, nosql","Candidate matches 12 required skill(s)..."
```

---

## How scoring works (short version)

```
final_score = 0.45 × TF-IDF cosine similarity
            + 0.40 × skill overlap with JD
            + 0.15 × experience match vs JD requirement
```

Full explanation, including known failure cases, is in
[`SCORING_METHOD.md`](SCORING_METHOD.md) — this is the required "note
explaining the scoring method" deliverable.

---

## Design decisions & tradeoffs

- **TF-IDF + cosine similarity instead of embeddings/LLM scoring.** Chosen
  because it's deterministic, free, runs fully offline, and is easy for a
  reviewer to audit — no API key needed to reproduce results. The tradeoff
  is that it's purely lexical: it won't recognize "built REST APIs" and
  "developed HTTP services" as similar phrasing the way an embedding model
  would. This is why skill-overlap (a more literal, curated match) is
  weighted almost as heavily.
- **Rule-based extraction by default, LLM optional.** Regex/keyword-based
  extraction for skills, experience, and contact info means the agent works
  with zero setup and zero cost, and every score is fully explainable. The
  `--use-llm` flag is there for teams who want richer natural-language
  reasoning per candidate, but the agent **never breaks** if the key is
  missing or the API call fails — it silently falls back to rule-based
  reasoning (see `src/reasoner.py`).
- **Curated skills vocabulary vs. free-form NLP.** A fixed list (~90 terms)
  keeps skill-matching precise and reviewable, at the cost of missing
  niche/uncommon skills not in the list. Easy to extend in
  `src/extractor.py`.
- **CSV + JSON output, no UI.** A CLI/backend was prioritized per the
  challenge's "clean backend or CLI is enough" guidance, so time went into
  a correct, well-tested pipeline instead of a frontend.
- **Mixed-format sample data on purpose.** The sample resumes include real
  `.pdf` and `.docx` files (not just `.txt`) specifically to prove the
  parser handles all three required formats, not just the easy case.

## What we'd improve with more time

See the "What we'd improve with more time" section at the bottom of
[`SCORING_METHOD.md`](SCORING_METHOD.md) — in short: semantic embedding
similarity, date-range-based experience calculation, LLM-assisted skill
extraction as a fallback for the curated vocabulary, and configurable
scoring weights per role family.

---

## Tech stack

- Python 3
- `pdfplumber` — PDF text extraction
- `python-docx` — DOCX text extraction
- `scikit-learn` — TF-IDF vectorization + cosine similarity
- `pandas` — CSV output
- `anthropic` (optional) — LLM-generated reasoning text
- `reportlab` — used only to generate the sample PDF resume in `data/`, not a runtime dependency of the agent itself
