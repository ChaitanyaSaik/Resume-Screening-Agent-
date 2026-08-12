"""
reasoner.py
-----------
Generates a short, human-readable explanation of WHY a candidate got their
score/rank.

Two modes:
  1. Rule-based (default, no API key needed) - deterministic sentence built
     from the matched/missing skills, experience, and education fields.
  2. LLM-powered (optional, --use-llm flag) - sends the JD + candidate
     summary to Claude and asks for a short natural-language justification.
     Falls back to rule-based automatically if no API key is configured or
     the API call fails, so the agent never breaks because of this step.
"""

import os


def rule_based_reasoning(candidate: dict) -> str:
    """Deterministic, explainable reasoning string. No API calls."""
    matched = candidate["matched_skills"]
    missing = candidate["missing_skills"]
    exp = candidate["experience_years"]
    req_exp = candidate["jd_required_years"]
    edu = candidate["education_level"]

    parts = []

    if matched:
        parts.append(f"matches {len(matched)} required skill(s) ({', '.join(matched[:6])}{'...' if len(matched) > 6 else ''})")
    else:
        parts.append("matches none of the explicitly listed required skills")

    if missing:
        parts.append(f"is missing {len(missing)} required skill(s) ({', '.join(missing[:6])}{'...' if len(missing) > 6 else ''})")

    if req_exp:
        if exp >= req_exp:
            parts.append(f"meets the {req_exp}+ year experience requirement with {exp} year(s)")
        else:
            parts.append(f"falls short of the {req_exp}+ year experience requirement (has {exp} year(s))")
    else:
        parts.append(f"has {exp} year(s) of stated experience")

    if edu != "Not specified":
        parts.append(f"holds a {edu}-level qualification")

    return "Candidate " + "; ".join(parts) + "."


def llm_reasoning(jd_text: str, candidate: dict) -> str:
    """
    Optional: ask Claude for a short, natural-language justification.
    Requires ANTHROPIC_API_KEY to be set in the environment / .env file.
    Silently falls back to rule-based reasoning on any failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return rule_based_reasoning(candidate)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are helping a recruiter understand a resume-screening result.

Job description (truncated):
{jd_text[:1200]}

Candidate summary:
- Matched required skills: {', '.join(candidate['matched_skills']) or 'none'}
- Missing required skills: {', '.join(candidate['missing_skills']) or 'none'}
- Years of experience: {candidate['experience_years']}
- Education level: {candidate['education_level']}
- Overall score: {candidate['final_score']}/100

In 2 sentences, explain why this candidate received this score and whether
they're a strong fit for the role. Be specific and concise. Do not repeat
the raw numbers back verbatim, write it like a recruiter note."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    except Exception as e:
        # Never let an LLM/network hiccup break the pipeline.
        return rule_based_reasoning(candidate) + f" [LLM reasoning unavailable: {e}]"


def generate_reasoning(jd_text: str, candidate: dict, use_llm: bool = False) -> str:
    if use_llm:
        return llm_reasoning(jd_text, candidate)
    return rule_based_reasoning(candidate)
