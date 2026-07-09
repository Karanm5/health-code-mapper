-- Crosswalk: (source_vocabulary, source_code) -> standard concept_id
-- Built from the OMOP vocabulary using the "Maps to" relationship, exactly as a
-- real OMOP ETL standardises source codes.
with source_concepts as (
    select
        concept_id      as source_concept_id,
        vocabulary_id   as source_vocabulary,
        concept_code    as source_code
    from {{ ref('stg_concept') }}
    where standard_concept is null or standard_concept <> 'S'
),
maps_to as (
    select concept_id_1 as source_concept_id, concept_id_2 as target_concept_id
    from {{ ref('stg_concept_relationship') }}
    where relationship_id = 'Maps to'
)
select
    s.source_vocabulary,
    s.source_code,
    s.source_concept_id,
    m.target_concept_id
from source_concepts s
join maps_to m using (source_concept_id)
