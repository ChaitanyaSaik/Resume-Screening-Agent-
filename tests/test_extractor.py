"""
tests/test_extractor.py
-------------------------
Unit tests for src/extractor.py -- skills, email, phone, experience,
education extraction.
"""

from src.extractor import (
    extract_skills,
    extract_email,
    extract_phone,
    extract_experience_years,
    extract_education_level,
    extract_candidate_profile,
)


def test_extract_skills_finds_known_skills():
    text = "Experienced in Python, Django, PostgreSQL and Docker."
    skills = extract_skills(text)
    assert "python" in skills
    assert "django" in skills
    assert "postgresql" in skills
    assert "docker" in skills


def test_extract_skills_is_case_insensitive():
    text = "PYTHON and JavaScript developer"
    skills = extract_skills(text)
    assert "python" in skills
    assert "javascript" in skills


def test_extract_skills_does_not_false_positive_on_substrings():
    # "r" is in the vocabulary as a language, must not match inside "director"
    text = "Worked as a director of engineering."
    skills = extract_skills(text)
    assert "r" not in skills


def test_extract_skills_returns_empty_set_for_no_matches():
    text = "This text has no recognizable technical skills in it at all."
    skills = extract_skills(text)
    assert isinstance(skills, set)


def test_extract_email_finds_valid_email():
    text = "Contact me at jane.doe@example.com for more info."
    assert extract_email(text) == "jane.doe@example.com"


def test_extract_email_returns_not_found_when_absent():
    text = "No contact info here."
    assert extract_email(text) == "Not found"


def test_extract_phone_handles_international_formats():
    cases = [
        "Phone: +91 98765 43210",
        "Call (415) 555-0132",
        "+234 802 555 0199",
        "+86 138 0013 8000",
        "+971 50 123 4567",
    ]
    for text in cases:
        result = extract_phone(text)
        assert result != "Not found", f"Failed to extract phone from: {text}"
        digit_count = sum(ch.isdigit() for ch in result)
        assert 7 <= digit_count <= 15


def test_extract_phone_returns_not_found_when_absent():
    text = "No phone number mentioned anywhere in this resume."
    assert extract_phone(text) == "Not found"


def test_extract_experience_years_basic():
    text = "I have 5 years of experience in software development."
    assert extract_experience_years(text) == 5


def test_extract_experience_years_with_plus_sign():
    text = "6+ years of experience building backend systems."
    assert extract_experience_years(text) == 6


def test_extract_experience_years_returns_zero_when_absent():
    text = "No experience duration mentioned."
    assert extract_experience_years(text) == 0


def test_extract_experience_years_takes_the_maximum():
    text = "2 years of experience as an intern, then 5 years of experience as a full-time engineer."
    assert extract_experience_years(text) == 5


def test_extract_education_level_detects_bachelor():
    text = "Bachelor of Technology in Computer Science"
    level, score = extract_education_level(text)
    assert level == "Bachelor"
    assert score == 3


def test_extract_education_level_detects_master_over_bachelor():
    text = "Bachelor of Science, followed by a Master of Science in Data Science"
    level, score = extract_education_level(text)
    assert level == "Master"
    assert score == 4


def test_extract_education_level_returns_not_specified_when_absent():
    text = "No education section in this text."
    level, score = extract_education_level(text)
    assert level == "Not specified"
    assert score == 0


def test_extract_candidate_profile_bundles_all_fields():
    text = (
        "John Doe\n"
        "john.doe@example.com | +1 415 555 0132\n"
        "5 years of experience in Python and Django.\n"
        "Bachelor of Technology in Computer Science."
    )
    profile = extract_candidate_profile("john_doe.txt", text)

    assert profile["filename"] == "john_doe.txt"
    assert profile["email"] == "john.doe@example.com"
    assert profile["phone"] != "Not found"
    assert "python" in profile["skills"]
    assert "django" in profile["skills"]
    assert profile["experience_years"] == 5
    assert profile["education_level"] == "Bachelor"
    assert profile["raw_text"] == text
