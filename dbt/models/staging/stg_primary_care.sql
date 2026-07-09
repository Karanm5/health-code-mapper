-- Primary care: Read v2 coded events
select
    patient_id                       as person_source_id,
    cast(event_date as date)         as event_date,
    'Read'                           as source_vocabulary,
    read_code                        as source_code
from read_csv_auto('{{ var("source_dir") }}/primary_care.csv', header=true)
