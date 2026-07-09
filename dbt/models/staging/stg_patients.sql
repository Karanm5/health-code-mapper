select
    patient_id                       as person_source_id,
    cast(birth_date as date)         as birth_date,
    sex
from read_csv_auto('{{ var("source_dir") }}/patients.csv', header=true)
