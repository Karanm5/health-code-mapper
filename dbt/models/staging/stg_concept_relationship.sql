select
    cast(concept_id_1 as bigint)     as concept_id_1,
    cast(concept_id_2 as bigint)     as concept_id_2,
    relationship_id
from read_csv_auto('{{ var("vocab_dir") }}/concept_relationship.csv', header=true)
