# Scoring Method

This note explains exactly how `final_score` (0–100) is calculated for each
resume, as required by the challenge deliverables.

## Overview

The score is a **weighted combination of three independent signals**. No
single signal decides the outcome, which reduces gaming (e.g. keyword
stuffing) and reduces the chance that one weak measurement sinks an
otherwise strong candidate.

| Component | Weight | What it measures |
|---|---|---|
| TF-IDF cosine similarity | 45% | Overall topical/textual overlap between the resume and the JD |
| Skill overlap | 40% | Fraction of the JD's explicitly required skills the resume literally contains |
| Experience match | 15% | How the candidate's stated years of experience compares to the JD's stated requirement |

`final_score = (0.45 * tfidf + 0.40 * skill_overlap + 0.15 * experience_match) * 100`

## 1. TF-IDF Cosine Similarity (`src/scorer.py`)

- The JD and every resume are vectorized together using **scikit-learn's
  `TfidfVectorizer`** (unigrams + bigrams, English stop-words removed), so
  the vocabulary is shared and scores are comparable across candidates.
- **Cosine similarity** is computed between the JD vector and each resume
  vector, producing a value in `[0, 1]`.
- This is a classic, fast, fully local (no API required) NLP similarity
  method. It captures general topical alignment — e.g. a resume full of
  backend/cloud/API language will score higher on a backend JD even if it
  doesn't hit every exact keyword.
- **Why TF-IDF instead of embeddings?** It's deterministic, doesn't need a
  model download or API key, is easy to explain to a non-technical
  reviewer, and is "good enough" at the JD-vs-resume scale this agent
  targets. See *Tradeoffs* below for what a production version would use
  instead.

## 2. Skill Overlap (`src/extractor.py` + `src/scorer.py`)

- A curated vocabulary of ~90 common technical/business skills
  (`SKILLS_VOCABULARY` in `extractor.py`) is matched against both the JD and
  each resume using word-boundary-safe regex matching (so `"r"` doesn't
  match inside `"director"`, etc.).
- `skill_overlap = |JD_skills ∩ resume_skills| / |JD_skills|`
- This is the most literal, explainable part of the score: "the JD asked
  for these 17 skills, this candidate has 12 of them."
- The agent also reports **matched_skills** and **missing_skills** per
  candidate so a recruiter can sanity-check the number immediately.

## 3. Experience Match (`src/scorer.py`)

- The JD is scanned for a stated minimum years-of-experience requirement
  (e.g. "3+ years") using a regex.
- The resume is scanned the same way for the candidate's stated years of
  experience.
- Scoring:
  - No requirement stated in JD → neutral `0.5` (doesn't help or hurt).
  - Candidate meets/exceeds the requirement → `1.0`.
  - Candidate falls short → partial credit, `candidate_years / required_years`.

## Tie-breaking / secondary signals

`education_level` and `education_score` are extracted and included in the
output for context and for a human reviewer to use as a tie-breaker, but
they **do not** currently affect `final_score` directly — education
requirements are too JD-specific (and easy to state as a hard filter
downstream) to bake into the core weight formula without more signal.

## Known failure cases / limitations

- **Skill vocabulary is finite.** A skill not in `SKILLS_VOCABULARY` (e.g.
  a niche internal tool name) will never be "matched," even if it's
  genuinely relevant. Fix: extend the vocabulary, or replace with an
  LLM-based skill extractor for production use.
- **TF-IDF has no semantic understanding.** "Built REST APIs" and
  "developed HTTP-based services" won't be recognized as similar by
  TF-IDF the way they would by an embedding model. This is the main
  reason skill-overlap is weighted almost as heavily as TF-IDF, to
  compensate with a more literal signal.
- **Experience years extraction is regex-based.** It looks for explicit
  "X years of experience"-style phrases. A resume that lists experience
  only via employment date ranges (e.g. "2019–Present") without ever
  stating "X years" will be scored as 0 years unless the ranges are summed
  — the sample resumes in `data/resumes/` intentionally state years
  explicitly to demonstrate the intended extraction; a production version
  should also parse date ranges.
- **No de-duplication / anti-gaming.** A resume that pastes the entire JD
  as invisible text would currently inflate its TF-IDF score. A production
  system should sanity-check resume structure (sections, formatting) before
  trusting extracted text.

## What we'd improve with more time

1. Swap/augment TF-IDF with sentence-embedding similarity (e.g.
   `sentence-transformers`) for genuine semantic matching.
2. Use the optional LLM reasoning path (`--use-llm`, see README) not just
   for the explanation text but to *extract* skills/experience more
   robustly than pure regex, with the regex output kept as a fast, free
   fallback.
3. Parse employment date ranges to compute total experience even when the
   resume never explicitly states "X years."
4. Add configurable weight profiles per role family (e.g. weight skills
   higher for a specialist role, weight TF-IDF higher for a generalist
   role).
