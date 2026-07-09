select
    cast(concept_id as bigint)       as concept_id,
    concept_name,
    domain_id,
    vocabulary_id,
    concept_class_id,
    standard_concept,
    cast(concept_code as varchar)    as concept_code
from read_csv_auto('{{ var("vocab_dir") }}/concept.csv', header=true, all_varchar=true)
