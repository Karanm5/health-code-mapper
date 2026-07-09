"""
Build a small, self-contained OMOP-style vocabulary for the demo.

This is NOT the full Athena vocabulary download (which is hundreds of MB and
requires an OHDSI/UMLS licence for SNOMED). It is a curated demonstration
subset shaped exactly like the real OMOP vocabulary tables so the pipeline
runs end-to-end with no external downloads. Swapping in the real Athena
vocabulary is a drop-in replacement: same table shapes, same column names.

Produces three CSVs under data/vocab/:
  - concept.csv                (standard SNOMED targets + non-standard ICD-10 / Read sources)
  - concept_relationship.csv   ("Maps to" ground-truth links, source -> standard)
  - source_codes.csv           (the raw source codes we want to map, as they'd arrive in EHR)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vocab")


@dataclass
class Entry:
    condition: str          # human label for the clinical idea
    snomed_id: int          # standard concept_id (SNOMED)
    snomed_name: str        # standard concept_name
    icd10_code: str         # ICD-10 source code (as in HES / death registry)
    icd10_name: str         # ICD-10 source term
    read_code: str          # Read v2 source code (as in primary care)
    read_name: str          # Read v2 source term


# Curated clinical concepts. Names are deliberately phrased *differently* across
# coding systems (as they really are), which is exactly what makes mapping hard
# and what the confidence layer is for.
ENTRIES = [
    Entry("Type 2 diabetes", 201826, "Type 2 diabetes mellitus",
          "E11", "Non-insulin-dependent diabetes mellitus",
          "C10F", "Type 2 diabetes mellitus"),
    Entry("Type 1 diabetes", 201254, "Type 1 diabetes mellitus",
          "E10", "Insulin-dependent diabetes mellitus",
          "C10E", "Type 1 diabetes mellitus"),
    Entry("Essential hypertension", 320128, "Essential hypertension",
          "I10", "Essential (primary) hypertension",
          "G20", "Essential hypertension"),
    Entry("Acute myocardial infarction", 4329847, "Myocardial infarction",
          "I21", "Acute myocardial infarction",
          "G30", "Acute myocardial infarction"),
    Entry("Atrial fibrillation", 313217, "Atrial fibrillation",
          "I48", "Atrial fibrillation and flutter",
          "G573", "Atrial fibrillation"),
    Entry("Heart failure", 316139, "Heart failure",
          "I50", "Heart failure",
          "G58", "Heart failure"),
    Entry("Asthma", 317009, "Asthma",
          "J45", "Asthma",
          "H33", "Asthma"),
    Entry("COPD", 255573, "Chronic obstructive pulmonary disease",
          "J44", "Other chronic obstructive pulmonary disease",
          "H3", "Chronic obstructive pulmonary disease"),
    Entry("Pneumonia", 255848, "Pneumonia",
          "J18", "Pneumonia, organism unspecified",
          "H2", "Pneumonia"),
    Entry("Stroke", 381316, "Cerebrovascular accident",
          "I63", "Cerebral infarction",
          "G66", "Cerebrovascular accident"),
    Entry("Chronic kidney disease", 46271022, "Chronic kidney disease",
          "N18", "Chronic kidney disease",
          "1Z1", "Chronic kidney disease"),
    Entry("Depressive disorder", 440383, "Depressive disorder",
          "F32", "Depressive episode",
          "E2B", "Depressive disorder"),
    Entry("Anxiety disorder", 442077, "Anxiety disorder",
          "F41", "Other anxiety disorders",
          "E200", "Anxiety states"),
    Entry("Osteoarthritis", 80180, "Osteoarthritis",
          "M19", "Other and unspecified osteoarthritis",
          "N05", "Osteoarthritis"),
    Entry("Rheumatoid arthritis", 80809, "Rheumatoid arthritis",
          "M06", "Other rheumatoid arthritis",
          "N040", "Rheumatoid arthritis"),
    Entry("Breast cancer", 4112853, "Malignant tumor of breast",
          "C50", "Malignant neoplasm of breast",
          "B34", "Malignant neoplasm of breast"),
    Entry("Lung cancer", 254637, "Non-small cell lung cancer",
          "C34", "Malignant neoplasm of bronchus and lung",
          "B22", "Malignant neoplasm of bronchus/lung"),
    Entry("Colorectal cancer", 443381, "Malignant tumor of colon",
          "C18", "Malignant neoplasm of colon",
          "B13", "Malignant neoplasm of colon"),
    Entry("Prostate cancer", 4163261, "Malignant tumor of prostate",
          "C61", "Malignant neoplasm of prostate",
          "B46", "Malignant neoplasm of prostate"),
    Entry("Epilepsy", 380378, "Epilepsy",
          "G40", "Epilepsy",
          "F25", "Epilepsy"),
    Entry("Parkinson disease", 381270, "Parkinson's disease",
          "G20", "Parkinson disease",
          "F12", "Parkinson's disease"),
    Entry("Dementia", 4182210, "Dementia",
          "F03", "Unspecified dementia",
          "E00", "Dementia"),
    Entry("Hypothyroidism", 140673, "Hypothyroidism",
          "E03", "Other hypothyroidism",
          "C04", "Hypothyroidism"),
    Entry("Gastro-oesophageal reflux", 318800, "Gastroesophageal reflux disease",
          "K21", "Gastro-oesophageal reflux disease",
          "J10", "Gastro-oesophageal reflux disease"),
    Entry("Chronic liver disease", 4064161, "Chronic liver disease",
          "K76", "Other diseases of liver",
          "J61", "Chronic liver disease"),
    Entry("Obesity", 433736, "Obesity",
          "E66", "Obesity",
          "C38", "Obesity"),
    Entry("Anaemia", 439777, "Anemia",
          "D64", "Other anaemias",
          "D0", "Anaemia"),
    Entry("Osteoporosis", 80502, "Osteoporosis",
          "M81", "Osteoporosis without pathological fracture",
          "N330", "Osteoporosis"),
    Entry("Migraine", 318736, "Migraine",
          "G43", "Migraine",
          "F26", "Migraine"),
    Entry("Psoriasis", 140168, "Psoriasis",
          "L40", "Psoriasis",
          "M11", "Psoriasis"),
]


def _write(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build() -> None:
    os.makedirs(OUT, exist_ok=True)

    concept_rows: list[list] = []
    rel_rows: list[list] = []
    source_rows: list[list] = []

    # Standard SNOMED targets get their real concept_id; source concepts get a
    # synthetic id range so nothing collides.
    src_id = 2_000_000_000

    for e in ENTRIES:
        # Standard target (SNOMED)
        concept_rows.append([
            e.snomed_id, e.snomed_name, "Condition", "SNOMED",
            "Clinical Finding", "S", e.snomed_id,
        ])
        # ICD-10 source concept
        icd_id = src_id
        src_id += 1
        concept_rows.append([
            icd_id, e.icd10_name, "Condition", "ICD10CM",
            "Diagnosis", "", e.icd10_code,
        ])
        rel_rows.append([icd_id, e.snomed_id, "Maps to"])
        source_rows.append(["ICD10CM", e.icd10_code, e.icd10_name, e.snomed_id])

        # Read v2 source concept
        read_id = src_id
        src_id += 1
        concept_rows.append([
            read_id, e.read_name, "Condition", "Read",
            "Read Code", "", e.read_code,
        ])
        rel_rows.append([read_id, e.snomed_id, "Maps to"])
        source_rows.append(["Read", e.read_code, e.read_name, e.snomed_id])

    _write(
        os.path.join(OUT, "concept.csv"),
        ["concept_id", "concept_name", "domain_id", "vocabulary_id",
         "concept_class_id", "standard_concept", "concept_code"],
        concept_rows,
    )
    _write(
        os.path.join(OUT, "concept_relationship.csv"),
        ["concept_id_1", "concept_id_2", "relationship_id"],
        rel_rows,
    )
    _write(
        os.path.join(OUT, "source_codes.csv"),
        ["source_vocabulary", "source_code", "source_name", "true_target_concept_id"],
        source_rows,
    )

    print(f"Wrote {len(concept_rows)} concepts, {len(rel_rows)} maps, "
          f"{len(source_rows)} source codes to {OUT}")


if __name__ == "__main__":
    build()
