"""
tests/test_scorer.py
----------------------
Unit tests for src/scorer.py -- TF-IDF similarity, skill overlap,
experience match, and the combined score_candidates pipeline.
"""

from src.scorer import (
    compute_tfidf_similarity,
    compute_skill_overlap,
    compute_experience_match,
    extract_jd_required_experience,
    score_candidates,
)
from src.extractor import extract_candidate_profile, extract_skills


def test_compute_tfidf_similarity_ranks_closer_text_higher():
    jd_text = "We need a Python backend developer with Django and PostgreSQL experience."
    resumes = [
        "Python developer skilled in Django and PostgreSQL, backend systems.",  # very close
        "Marketing coordinator with social media and content strategy skills.",  # unrelated
    ]
    scores = compute_tfidf_similarity(jd_text, resumes)
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_compute_tfidf_similarity_returns_values_in_valid_range():
    jd_text = "Python developer needed."
    resumes = ["Python developer with experience.", "Completely unrelated text about gardening."]
    scores = compute_tfidf_similarity(jd_text, resumes)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_compute_skill_overlap_full_match():
    jd_skills = {"python", "django", "postgresql"}
    resume_skills = {"python", "django", "postgresql", "docker"}
    assert compute_skill_overlap(jd_skills, resume_skills) == 1.0


def test_compute_skill_overlap_partial_match():
    jd_skills = {"python", "django", "postgresql", "docker"}
    resume_skills = {"python", "django"}
    assert compute_skill_overlap(jd_skills, resume_skills) == 0.5


def test_compute_skill_overlap_no_match():
    jd_skills = {"python", "django"}
    resume_skills = {"javascript", "react"}
    assert compute_skill_overlap(jd_skills, resume_skills) == 0.0


def test_compute_skill_overlap_empty_jd_skills_returns_zero():
    assert compute_skill_overlap(set(), {"python"}) == 0.0


def test_extract_jd_required_experience():
    jd_text = "We require 3+ years of experience in backend development."
    assert extract_jd_required_experience(jd_text) == 3


def test_extract_jd_required_experience_absent_returns_zero():
    jd_text = "No specific experience requirement stated."
    assert extract_jd_required_experience(jd_text) == 0


def test_compute_experience_match_meets_requirement():
    assert compute_experience_match(required_years=3, candidate_years=5) == 1.0


def test_compute_experience_match_exact_requirement():
    assert compute_experience_match(required_years=3, candidate_years=3) == 1.0


def test_compute_experience_match_falls_short():
    result = compute_experience_match(required_years=4, candidate_years=2)
    assert result == 0.5


def test_compute_experience_match_no_requirement_is_neutral():
    assert compute_experience_match(required_years=0, candidate_years=0) == 0.5


def test_score_candidates_ranks_stronger_candidate_higher():
    jd_text = "Backend Python Developer. Requires 3+ years of experience with Python, Django, PostgreSQL, and Docker."
    jd_skills = extract_skills(jd_text)

    strong_resume = (
        "Senior backend engineer with 5 years of experience in Python, "
        "Django, PostgreSQL, and Docker, building REST APIs."
    )
    weak_resume = (
        "Marketing coordinator with 2 years of experience in social media "
        "and content strategy. No technical background."
    )

    candidates = [
        extract_candidate_profile("strong.txt", strong_resume),
        extract_candidate_profile("weak.txt", weak_resume),
    ]

    scored = score_candidates(jd_text, jd_skills, candidates)
    scored.sort(key=lambda c: c["final_score"], reverse=True)

    assert scored[0]["filename"] == "strong.txt"
    assert scored[0]["final_score"] > scored[1]["final_score"]
    assert 0 <= scored[0]["final_score"] <= 100
    assert 0 <= scored[1]["final_score"] <= 100


def test_score_candidates_includes_matched_and_missing_skills():
    jd_text = "Requires Python, Django, and Kubernetes. 2+ years of experience."
    jd_skills = extract_skills(jd_text)

    resume_text = "Python developer with 3 years of experience. Knows Django well."
    candidates = [extract_candidate_profile("candidate.txt", resume_text)]

    scored = score_candidates(jd_text, jd_skills, candidates)
    candidate = scored[0]

    assert "python" in candidate["matched_skills"]
    assert "django" in candidate["matched_skills"]
    assert "kubernetes" in candidate["missing_skills"]
