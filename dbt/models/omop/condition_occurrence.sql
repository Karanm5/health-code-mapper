-- Standardised conditions from both HES (ICD-10) and primary care (Read).
with events as (
    select * from {{ ref('stg_hes') }}
    union all
    select * from {{ ref('stg_primary_care') }}
)
select
    row_number() over (order by e.person_source_id, e.event_date) as condition_occurrence_id,
    e.person_source_id           as person_id,
    x.target_concept_id          as condition_concept_id,
    e.event_date                 as condition_start_date,
    x.source_concept_id          as condition_source_concept_id,
    e.source_code                as condition_source_value,
    e.source_vocabulary          as source_vocabulary
from events e
join {{ ref('xref_source_to_standard') }} x
  on e.source_vocabulary = x.source_vocabulary
 and e.source_code       = x.source_code
