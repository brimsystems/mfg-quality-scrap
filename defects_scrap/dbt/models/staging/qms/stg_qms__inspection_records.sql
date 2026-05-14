with source as (

    select * from {{ source('qms', 'inspection_records') }}

),

staged as (

    select
        inspection_id,
        work_order_id,
        cast(inspection_date as timestamp)      as inspection_date,
        inspector_id,
        cast(quantity_inspected as integer)     as quantity_inspected,
        cast(quantity_passed as integer)        as quantity_passed,
        cast(quantity_failed as integer)        as quantity_failed,

        -- Defect code: raw field contains free-text variants and typos.
        -- Clean field contains the canonical value from the controlled vocabulary.
        -- Both preserved — raw for audit, clean for analysis.
        defect_code_clean                       as defect_code,
        defect_code_raw,

        disposition,
        notes,

        -- Flag records with timestamps outside shift windows.
        -- These are retroactive entries and should not be used for
        -- time-of-day analysis without review.
        case
            when cast(inspection_date as timestamp)::time
                 not between '06:00:00' and '21:59:59'
            then true
            else false
        end                                     as is_anomalous_timestamp

    from source

),

deduplicated as (

    -- The QMS creates duplicate inspection records when a session times out
    -- during entry and the inspector re-submits. Duplicates share work_order_id
    -- but have distinct inspection_ids. The earliest record is retained as the
    -- authoritative entry; subsequent records are discarded.
    select *
    from staged
    qualify row_number() over (
        partition by work_order_id
        order by inspection_id asc
    ) = 1

)

select * from deduplicated