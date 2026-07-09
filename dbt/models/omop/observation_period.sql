with all_events as (
    select person_source_id, event_date as d from {{ ref('stg_hes') }}
    union all
    select person_source_id, event_date from {{ ref('stg_primary_care') }}
    union all
    select person_source_id, death_date from {{ ref('stg_deaths') }}
)
select
    row_number() over (order by person_source_id)   as observation_period_id,
    person_source_id                                as person_id,
    min(d)                                          as observation_period_start_date,
    max(d)                                          as observation_period_end_date
from all_events
group by person_source_id
