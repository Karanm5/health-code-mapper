-- Hospital Episode Statistics: ICD-10 coded diagnoses
select
    patient_id                       as person_source_id,
    cast(admission_date as date)     as event_date,
    'ICD10CM'                        as source_vocabulary,
    icd10_code                       as source_code
from read_csv_auto('{{ var("source_dir") }}/hes_diagnoses.csv', header=true)
