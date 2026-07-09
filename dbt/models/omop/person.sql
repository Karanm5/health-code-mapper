select
    person_source_id                                 as person_id,
    case sex when 'M' then 8507 when 'F' then 8532 else 0 end as gender_concept_id,
    extract(year  from birth_date)                   as year_of_birth,
    extract(month from birth_date)                   as month_of_birth,
    extract(day   from birth_date)                   as day_of_birth,
    sex                                              as gender_source_value
from {{ ref('stg_patients') }}
