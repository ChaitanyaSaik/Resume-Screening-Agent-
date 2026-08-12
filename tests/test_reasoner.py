"""
tests/test_reasoner.py
------------------------
Unit tests for src/reasoner.py -- rule-based reasoning generation and
the LLM-reasoning fallback behavior (never crashes without an API key).
"""

import os
import pytest

from src.reasoner import rule_based_reasoning, generate_reasoning


def make_candidate(
    matched_skills=None,
    missing_skills=None,
    experience_years=3,
    jd_required_years=3,
    education_level="Bachelor",
):
    return {
        "matched_skills": matched_skills or [],
        "missing_skills": missing_skills or [],
        "experience_years": experience_years,
        "jd_required_years": jd_required_years,
        "education_level": education_level,
    }


def test_rule_based_reasoning_mentions_matched_skills():
    candidate = make_candidate(matched_skills=["python", "django"])
    reasoning = rule_based_reasoning(candidate)
    assert "matches 2 required skill(s)" in reasoning
    assert "python" in reasoning


def test_rule_based_reasoning_mentions_missing_skills():
    candidate = make_candidate(missing_skills=["docker", "kubernetes"])
    reasoning = rule_based_reasoning(candidate)
    assert "missing 2 required skill(s)" in reasoning


def test_rule_based_reasoning_no_matched_skills_phrasing():
    candidate = make_candidate(matched_skills=[])
    reasoning = rule_based_reasoning(candidate)
    assert "matches none of the explicitly listed required skills" in reasoning


def test_rule_based_reasoning_meets_experience_requirement():
    candidate = make_candidate(experience_years=5, jd_required_years=3)
    reasoning = rule_based_reasoning(candidate)
    assert "meets the 3+ year experience requirement" in reasoning


def test_rule_based_reasoning_falls_short_of_experience():
    candidate = make_candidate(experience_years=1, jd_required_years=3)
    reasoning = rule_based_reasoning(candidate)
    assert "falls short of the 3+ year experience requirement" in reasoning


def test_rule_based_reasoning_mentions_education_when_specified():
    candidate = make_candidate(education_level="Master")
    reasoning = rule_based_reasoning(candidate)
    assert "Master-level qualification" in reasoning


def test_rule_based_reasoning_omits_education_when_not_specified():
    candidate = make_candidate(education_level="Not specified")
    reasoning = rule_based_reasoning(candidate)
    assert "qualification" not in reasoning


def test_generate_reasoning_defaults_to_rule_based():
    candidate = make_candidate(matched_skills=["python"])
    reasoning = generate_reasoning("some jd text", candidate, use_llm=False)
    assert isinstance(reasoning, str)
    assert len(reasoning) > 0


def test_generate_reasoning_with_llm_falls_back_without_api_key(monkeypatch):
    """
    Critical robustness test: if --use-llm is passed but no ANTHROPIC_API_KEY
    is set, the agent must NOT crash -- it should silently fall back to
    rule-based reasoning.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    candidate = make_candidate(matched_skills=["python"])

    reasoning = generate_reasoning("some jd text", candidate, use_llm=True)

    assert isinstance(reasoning, str)
    assert len(reasoning) > 0
    # Should equal the plain rule-based output since no key was available
    assert reasoning == rule_based_reasoning(candidate)
