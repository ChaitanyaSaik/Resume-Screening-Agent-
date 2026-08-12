"""
extractor.py
------------
Pulls structured signals out of raw resume/JD text:
  - skills (matched against a curated skills vocabulary)
  - years of experience
  - education level
  - contact info (email/phone) -- resume only

This uses fast, transparent rule-based extraction (regex + keyword
matching) rather than an LLM call, so the agent works even with
zero API keys and the scoring is fully explainable/reproducible.
"""

import re

# A reasonably broad, curated vocabulary of tech/business skills.
# Extend this list to tune the agent for other domains.
SKILLS_VOCABULARY = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "sql", "r", "scala", "kotlin", "swift", "php", "ruby",
    # Web / Backend
    "django", "flask", "fastapi", "react", "angular", "vue", "node.js", "nodejs",
    "express", "spring boot", "rest api", "graphql", "microservices",
    # Data / ML
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "keras", "nlp", "machine learning", "deep learning", "data analysis",
    "data visualization", "power bi", "tableau", "matplotlib", "llm",
    "generative ai", "computer vision", "opencv",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "jenkins", "linux", "git", "github actions",
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "nosql",
    # Soft / Business skills
    "project management", "agile", "scrum", "communication", "leadership",
    "stakeholder management", "team management", "problem solving",
    # Other common
    "excel", "jira", "figma", "html", "css", "api integration", "testing",
    "unit testing", "cybersecurity", "networking", "android", "ios",
]

EDUCATION_KEYWORDS = {
    "phd": 5, "doctorate": 5,
    "master": 4, "m.tech": 4, "mtech": 4, "mba": 4, "m.sc": 4, "msc": 4,
    "bachelor": 3, "b.tech": 3, "btech": 3, "b.e": 3, "b.sc": 3, "bsc": 3,
    "diploma": 2,
    "high school": 1,
}

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# Matches international/local phone formats such as:
# +91 98765 43210 | (555) 123-4567 | +234 802 555 0199 | +86 138 0013 8000
PHONE_REGEX = re.compile(
    r"(?<!\d)(\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?(?:[-.\s]\d{2,5}){1,3}(?!\d)"
)

# Matches things like "5 years of experience", "3+ years", "experience: 4 yrs"
EXPERIENCE_REGEX = re.compile(
    r"(\d{1,2})\+?\s*(?:years|yrs|year)\s*(?:of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)


def extract_skills(text: str) -> set:
    """Return the set of vocabulary skills found in the text (case-insensitive)."""
    text_lower = text.lower()
    found = set()
    for skill in SKILLS_VOCABULARY:
        # Word-boundary-ish match so "r" doesn't match inside "director"
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def extract_email(text: str) -> str:
    match = EMAIL_REGEX.search(text)
    return match.group(0) if match else "Not found"


def extract_phone(text: str) -> str:
    """Find the first candidate match that has a plausible number of digits (7-15)."""
    for match in PHONE_REGEX.finditer(text):
        candidate = match.group(0).strip()
        digit_count = sum(ch.isdigit() for ch in candidate)
        if 7 <= digit_count <= 15:
            return candidate
    return "Not found"


def extract_experience_years(text: str) -> int:
    """
    Best-effort extraction of years of experience.
    Looks for explicit patterns like '5 years of experience'.
    Falls back to 0 if nothing is found (does not guess).
    """
    matches = EXPERIENCE_REGEX.findall(text)
    years = [int(m) for m in matches if m.isdigit()]
    return max(years) if years else 0


def extract_education_level(text: str) -> tuple:
    """
    Returns (level_name, level_score) for the highest education mentioned.
    level_score is used for lightweight ranking tie-breaks.
    """
    text_lower = text.lower()
    best_level, best_score = "Not specified", 0
    for keyword, score in EDUCATION_KEYWORDS.items():
        if keyword in text_lower and score > best_score:
            best_level, best_score = keyword.title(), score
    return best_level, best_score


def extract_candidate_profile(filename: str, text: str) -> dict:
    """Bundle all extracted fields for one resume into a single dict."""
    return {
        "filename": filename,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education_level": extract_education_level(text)[0],
        "education_score": extract_education_level(text)[1],
        "raw_text": text,
    }
