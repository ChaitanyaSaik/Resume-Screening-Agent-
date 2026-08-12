"""
scorer.py
---------
Computes a relevance score for each resume against the job description.

Method (see SCORING_METHOD.md for the full writeup):
  1. Semantic/textual similarity  -> TF-IDF vectorization + cosine similarity
     between the JD and each resume's full text. Captures overall topical
     overlap, not just exact keyword hits.
  2. Skill overlap score          -> fraction of JD-required skills that are
     present in the resume, using the curated skills vocabulary. Rewards
     resumes that literally have the skills the JD asks for.
  3. Experience match score       -> how close the candidate's years of
     experience are to what the JD asks for (if the JD states a requirement).

Final score = weighted sum of the three, scaled to 0-100.
Weights are configurable via SCORE_WEIGHTS below.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCORE_WEIGHTS = {
    "tfidf_similarity": 0.45,
    "skill_overlap": 0.40,
    "experience_match": 0.15,
}

JD_EXPERIENCE_REGEX = re.compile(
    r"(\d{1,2})\+?\s*(?:years|yrs|year)\s*(?:of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)


def compute_tfidf_similarity(jd_text: str, resume_texts: list) -> list:
    """
    Vectorize the JD + all resumes together (shared vocabulary), then
    compute cosine similarity of each resume against the JD.
    Returns a list of similarity scores in [0, 1], same order as resume_texts.
    """
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return similarities.tolist()


def compute_skill_overlap(jd_skills: set, resume_skills: set) -> float:
    """
    Fraction of JD-required skills that appear in the resume.
    Returns 0.0 if the JD lists no recognizable skills (avoids div-by-zero).
    """
    if not jd_skills:
        return 0.0
    matched = jd_skills.intersection(resume_skills)
    return len(matched) / len(jd_skills)


def extract_jd_required_experience(jd_text: str) -> int:
    """Best-effort extraction of the minimum years of experience the JD asks for."""
    matches = JD_EXPERIENCE_REGEX.findall(jd_text)
    years = [int(m) for m in matches if m.isdigit()]
    return max(years) if years else 0


def compute_experience_match(required_years: int, candidate_years: int) -> float:
    """
    1.0 if candidate meets/exceeds requirement.
    Otherwise a partial score proportional to how close they are.
    If the JD states no requirement, return a neutral 0.5 (doesn't hurt/help much).
    """
    if required_years == 0:
        return 0.5
    if candidate_years >= required_years:
        return 1.0
    return max(0.0, candidate_years / required_years)


def score_candidates(jd_text: str, jd_skills: set, candidates: list) -> list:
    """
    Takes the JD text/skills and a list of candidate profile dicts
    (from extractor.extract_candidate_profile), and returns the same
    list with score fields added:
        tfidf_score, skill_score, experience_score, final_score,
        matched_skills, missing_skills
    """
    resume_texts = [c["raw_text"] for c in candidates]
    tfidf_scores = compute_tfidf_similarity(jd_text, resume_texts)
    required_years = extract_jd_required_experience(jd_text)

    for candidate, tfidf_score in zip(candidates, tfidf_scores):
        skill_score = compute_skill_overlap(jd_skills, candidate["skills"])
        exp_score = compute_experience_match(required_years, candidate["experience_years"])

        final_score = (
            SCORE_WEIGHTS["tfidf_similarity"] * tfidf_score
            + SCORE_WEIGHTS["skill_overlap"] * skill_score
            + SCORE_WEIGHTS["experience_match"] * exp_score
        ) * 100

        candidate["tfidf_score"] = round(tfidf_score * 100, 2)
        candidate["skill_score"] = round(skill_score * 100, 2)
        candidate["experience_score"] = round(exp_score * 100, 2)
        candidate["final_score"] = round(final_score, 2)
        candidate["matched_skills"] = sorted(jd_skills.intersection(candidate["skills"]))
        candidate["missing_skills"] = sorted(jd_skills - candidate["skills"])
        candidate["jd_required_years"] = required_years

    return candidates
