"""Smoke tests: the synthetic data generators produce well-formed files."""
import csv
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_vocabulary_builds(tmp_path):
    subprocess.run([sys.executable, "data/build_vocabulary.py"], cwd=ROOT, check=True)
    with open(os.path.join(ROOT, "data/vocab/source_codes.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {"source_vocabulary", "source_code", "true_target_concept_id"} <= set(rows[0])


def test_source_generates():
    subprocess.run([sys.executable, "data/generate_source.py"], cwd=ROOT, check=True)
    for name in ["patients.csv", "hes_diagnoses.csv", "primary_care.csv", "deaths.csv"]:
        path = os.path.join(ROOT, "data/source", name)
        assert os.path.exists(path) and os.path.getsize(path) > 0
