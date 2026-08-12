"""
tests/test_integration.py
---------------------------
End-to-end test: runs the full agent (main.run) against the real sample
JD + sample resumes shipped in data/, exactly like a reviewer running
`python main.py` would, and checks the output files and ranking make sense.
"""

import json
import os

import pandas as pd
import pytest

from main import run

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_JD = os.path.join(PROJECT_ROOT, "data", "job_description.txt")
SAMPLE_RESUMES_DIR = os.path.join(PROJECT_ROOT, "data", "resumes")


@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path / "test_output")


def test_full_pipeline_runs_end_to_end_and_writes_outputs(output_dir):
    candidates = run(
        jd_path=SAMPLE_JD,
        resumes_dir=SAMPLE_RESUMES_DIR,
        out_dir=output_dir,
        use_llm=False,
        top_n=5,
    )

    # At least 10 sample resumes are expected per the challenge requirements
    assert len(candidates) >= 10

    csv_path = os.path.join(output_dir, "ranked_candidates.csv")
    json_path = os.path.join(output_dir, "ranked_candidates.json")
    assert os.path.exists(csv_path)
    assert os.path.exists(json_path)


def test_pipeline_output_is_sorted_descending_by_score(output_dir):
    candidates = run(
        jd_path=SAMPLE_JD,
        resumes_dir=SAMPLE_RESUMES_DIR,
        out_dir=output_dir,
        use_llm=False,
        top_n=5,
    )

    scores = [c["final_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_ranks_strong_backend_candidate_above_non_technical_candidate(output_dir):
    """
    Sanity check on real sample data: a senior backend engineer resume
    should clearly outrank a marketing-only resume for a backend JD.
    """
    candidates = run(
        jd_path=SAMPLE_JD,
        resumes_dir=SAMPLE_RESUMES_DIR,
        out_dir=output_dir,
        use_llm=False,
        top_n=len(os.listdir(SAMPLE_RESUMES_DIR)),
    )

    by_filename = {c["filename"]: c for c in candidates}

    # resume_03 = senior backend engineer (strong fit)
    # resume_12 = marketing coordinator, no technical background (poor fit)
    backend_candidate = next(c for name, c in by_filename.items() if "daniel_okafor" in name)
    marketing_candidate = next(c for name, c in by_filename.items() if "olivia_brown" in name)

    assert backend_candidate["final_score"] > marketing_candidate["final_score"]


def test_pipeline_csv_output_has_expected_columns(output_dir):
    run(
        jd_path=SAMPLE_JD,
        resumes_dir=SAMPLE_RESUMES_DIR,
        out_dir=output_dir,
        use_llm=False,
        top_n=5,
    )

    df = pd.read_csv(os.path.join(output_dir, "ranked_candidates.csv"))
    expected_columns = {
        "rank", "filename", "final_score", "tfidf_score", "skill_score",
        "experience_score", "experience_years", "education_level",
        "email", "phone", "matched_skills", "missing_skills", "reasoning",
    }
    assert expected_columns.issubset(set(df.columns))
    assert len(df) >= 10


def test_pipeline_json_output_is_valid_and_matches_csv_count(output_dir):
    run(
        jd_path=SAMPLE_JD,
        resumes_dir=SAMPLE_RESUMES_DIR,
        out_dir=output_dir,
        use_llm=False,
        top_n=5,
    )

    with open(os.path.join(output_dir, "ranked_candidates.json")) as f:
        data = json.load(f)

    df = pd.read_csv(os.path.join(output_dir, "ranked_candidates.csv"))

    assert isinstance(data, list)
    assert len(data) == len(df)
    # raw_text should be stripped out of the JSON for readability
    assert "raw_text" not in data[0]


def test_pipeline_handles_mixed_file_formats(output_dir):
    """
    data/resumes intentionally contains .txt, .pdf, and .docx files.
    Confirms all three formats were actually parsed (not silently skipped).
    """
    candidates = run(
        jd_path=SAMPLE_JD,
        resumes_dir=SAMPLE_RESUMES_DIR,
        out_dir=output_dir,
        use_llm=False,
        top_n=5,
    )

    extensions_found = {os.path.splitext(c["filename"])[1] for c in candidates}
    assert ".txt" in extensions_found
    assert ".pdf" in extensions_found
    assert ".docx" in extensions_found


def test_pipeline_raises_clean_error_for_missing_jd_file(output_dir):
    with pytest.raises(SystemExit):
        run(
            jd_path="data/does_not_exist.txt",
            resumes_dir=SAMPLE_RESUMES_DIR,
            out_dir=output_dir,
            use_llm=False,
            top_n=5,
        )


def test_pipeline_raises_clean_error_for_missing_resumes_folder(output_dir):
    with pytest.raises(SystemExit):
        run(
            jd_path=SAMPLE_JD,
            resumes_dir="data/does_not_exist_folder",
            out_dir=output_dir,
            use_llm=False,
            top_n=5,
        )
