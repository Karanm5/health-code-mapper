select
    d.person_source_id     as person_id,
    d.death_date           as death_date,
    x.target_concept_id    as cause_concept_id,
    d.source_code          as cause_source_value
from {{ ref('stg_deaths') }} d
left join {{ ref('xref_source_to_standard') }} x
  on d.source_vocabulary = x.source_vocabulary
 and d.source_code       = x.source_code
