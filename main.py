"""
main.py
-------
Resume Screening Agent - entry point.

Usage:
    python main.py --jd data/job_description.txt --resumes data/resumes --out output

    Optional flags:
      --use-llm     Use Claude to generate the reasoning text for each
                     candidate (requires ANTHROPIC_API_KEY). Without this
                     flag, reasoning is fully rule-based and needs no API key.
      --top N       Only print the top N candidates to the console (default 10).

See README.md for full setup instructions.
"""

import argparse
import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser import extract_text, load_resumes
from src.extractor import extract_candidate_profile, extract_skills
from src.scorer import score_candidates
from src.reasoner import generate_reasoning


def run(jd_path: str, resumes_dir: str, out_dir: str, use_llm: bool, top_n: int):
    load_dotenv()  # picks up ANTHROPIC_API_KEY from a .env file if present

    print(f"\nResume Screening Agent")
    print(f"=======================")
    print(f"Job description : {jd_path}")
    print(f"Resumes folder   : {resumes_dir}")
    print(f"LLM reasoning    : {'ON (Claude)' if use_llm else 'OFF (rule-based)'}\n")

    # 1. Load JD
    if not os.path.exists(jd_path):
        print(f"ERROR: Job description file not found: {jd_path}")
        sys.exit(1)
    jd_text = extract_text(jd_path)
    jd_skills = extract_skills(jd_text)
    print(f"[1/4] Loaded job description ({len(jd_text)} chars). "
          f"Detected {len(jd_skills)} required skill(s): {sorted(jd_skills)}\n")

    # 2. Load & parse resumes
    if not os.path.isdir(resumes_dir):
        print(f"ERROR: Resumes folder not found: {resumes_dir}")
        sys.exit(1)
    raw_resumes = load_resumes(resumes_dir)
    if not raw_resumes:
        print("ERROR: No parseable resumes found (.pdf, .docx, .txt).")
        sys.exit(1)
    print(f"[2/4] Parsed {len(raw_resumes)} resume(s).\n")

    # 3. Extract structured profiles
    candidates = [
        extract_candidate_profile(filename, text)
        for filename, text in raw_resumes.items()
    ]

    # 4. Score
    candidates = score_candidates(jd_text, jd_skills, candidates)
    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    print(f"[3/4] Scored and ranked {len(candidates)} candidate(s).\n")

    # 5. Reasoning
    for candidate in candidates:
        candidate["reasoning"] = generate_reasoning(jd_text, candidate, use_llm=use_llm)
    print(f"[4/4] Generated reasoning for each candidate.\n")

    # 6. Output
    os.makedirs(out_dir, exist_ok=True)
    write_outputs(candidates, out_dir)

    # 7. Console summary
    print_summary(candidates, top_n)

    return candidates


def write_outputs(candidates: list, out_dir: str):
    # JSON output (full detail, sets converted to sorted lists)
    json_ready = []
    for c in candidates:
        entry = dict(c)
        entry["skills"] = sorted(entry["skills"])
        entry.pop("raw_text", None)  # keep JSON readable
        json_ready.append(entry)

    json_path = os.path.join(out_dir, "ranked_candidates.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2)

    # CSV output (summary columns)
    df = pd.DataFrame([
        {
            "rank": i + 1,
            "filename": c["filename"],
            "final_score": c["final_score"],
            "tfidf_score": c["tfidf_score"],
            "skill_score": c["skill_score"],
            "experience_score": c["experience_score"],
            "experience_years": c["experience_years"],
            "education_level": c["education_level"],
            "email": c["email"],
            "phone": c["phone"],
            "matched_skills": ", ".join(c["matched_skills"]),
            "missing_skills": ", ".join(c["missing_skills"]),
            "reasoning": c["reasoning"],
        }
        for i, c in enumerate(candidates)
    ])
    csv_path = os.path.join(out_dir, "ranked_candidates.csv")
    df.to_csv(csv_path, index=False)

    print(f"Outputs written:\n  - {csv_path}\n  - {json_path}\n")


def print_summary(candidates: list, top_n: int):
    print(f"Top {min(top_n, len(candidates))} Candidates")
    print("=" * 60)
    for i, c in enumerate(candidates[:top_n], start=1):
        print(f"{i}. {c['filename']}  —  Score: {c['final_score']}/100")
        print(f"   Skills matched : {', '.join(c['matched_skills']) or 'none'}")
        print(f"   Skills missing : {', '.join(c['missing_skills']) or 'none'}")
        print(f"   Experience     : {c['experience_years']} yr(s) "
              f"(JD wants {c['jd_required_years']}+)")
        print(f"   Education      : {c['education_level']}")
        print(f"   Reasoning      : {c['reasoning']}")
        print("-" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--jd", default="data/job_description.txt",
                         help="Path to job description file (.txt/.pdf/.docx)")
    parser.add_argument("--resumes", default="data/resumes",
                         help="Path to folder containing resumes")
    parser.add_argument("--out", default="output",
                         help="Output folder for ranked_candidates.csv/json")
    parser.add_argument("--use-llm", action="store_true",
                         help="Use Claude to generate reasoning text (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--top", type=int, default=10,
                         help="Number of top candidates to print to console")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.jd, args.resumes, args.out, args.use_llm, args.top)
