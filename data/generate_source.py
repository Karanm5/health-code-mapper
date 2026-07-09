"""
Generate synthetic, UK-Biobank-shaped linked source EHR data.

UK Biobank delivers health outcomes as *linked* records from separate sources,
each with its own coding system:
  - Hospital Episode Statistics (HES)  -> ICD-10 diagnoses
  - Primary care                       -> Read v2 clinical codes
  - Death registry                     -> ICD-10 cause of death

This script reproduces that structure with entirely synthetic patients so the
OMOP ETL has realistic, multi-source, multi-vocabulary input to harmonise.
No real patient data is used or required.

Outputs under data/source/:
  - patients.csv
  - hes_diagnoses.csv      (ICD-10)
  - primary_care.csv       (Read v2)
  - deaths.csv             (ICD-10 cause)
"""
from __future__ import annotations

import csv
import os
import random
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = os.path.join(HERE, "vocab")
OUT = os.path.join(HERE, "source")

N_PATIENTS = 2000
SEED = 42


def _load_source_codes() -> dict[str, list[tuple[str, str]]]:
    """Return {'ICD10CM': [(code, name), ...], 'Read': [...]} from the vocab."""
    by_vocab: dict[str, list[tuple[str, str]]] = {"ICD10CM": [], "Read": []}
    with open(os.path.join(VOCAB, "source_codes.csv")) as f:
        for row in csv.DictReader(f):
            by_vocab[row["source_vocabulary"]].append(
                (row["source_code"], row["source_name"])
            )
    return by_vocab


def _rand_date(start: date, end: date, rng: random.Random) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def generate() -> None:
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    codes = _load_source_codes()

    patients = []
    for pid in range(1, N_PATIENTS + 1):
        birth = _rand_date(date(1937, 1, 1), date(1970, 12, 31), rng)
        sex = rng.choice(["M", "F"])
        patients.append((pid, birth.isoformat(), sex))

    with open(os.path.join(OUT, "patients.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "birth_date", "sex"])
        w.writerows(patients)

    # HES: ICD-10 hospital diagnoses (each patient has 0-4 episodes)
    hes = []
    for pid, birth, _sex in patients:
        for _ in range(rng.randint(0, 4)):
            code, _name = rng.choice(codes["ICD10CM"])
            admit = _rand_date(date(2006, 1, 1), date(2023, 12, 31), rng)
            hes.append((pid, admit.isoformat(), code))
    with open(os.path.join(OUT, "hes_diagnoses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "admission_date", "icd10_code"])
        w.writerows(hes)

    # Primary care: Read v2 codes (each patient has 0-6 events)
    pc = []
    for pid, birth, _sex in patients:
        for _ in range(rng.randint(0, 6)):
            code, _name = rng.choice(codes["Read"])
            event = _rand_date(date(2006, 1, 1), date(2023, 12, 31), rng)
            pc.append((pid, event.isoformat(), code))
    with open(os.path.join(OUT, "primary_care.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "event_date", "read_code"])
        w.writerows(pc)

    # Deaths: ~12% of patients, ICD-10 cause
    deaths = []
    for pid, birth, _sex in patients:
        if rng.random() < 0.12:
            code, _name = rng.choice(codes["ICD10CM"])
            dod = _rand_date(date(2010, 1, 1), date(2023, 12, 31), rng)
            deaths.append((pid, dod.isoformat(), code))
    with open(os.path.join(OUT, "deaths.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "death_date", "icd10_cause"])
        w.writerows(deaths)

    print(f"Generated {len(patients)} patients | {len(hes)} HES rows | "
          f"{len(pc)} primary-care rows | {len(deaths)} deaths -> {OUT}")


if __name__ == "__main__":
    generate()
