select
    patient_id                       as person_source_id,
    cast(death_date as date)         as death_date,
    'ICD10CM'                        as source_vocabulary,
    icd10_cause                      as source_code
from read_csv_auto('{{ var("source_dir") }}/deaths.csv', header=true)
