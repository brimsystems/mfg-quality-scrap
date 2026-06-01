with source as (

    select * from {{ source('qms', 'scrap_events') }}

),

staged as (

    select
        scrap_id,
        work_order_id,
        inspection_id,
        cast(scrap_date as timestamp)               as scrap_date,
        machine_id,
        operator_id,
        shift_code,
        material_type,
        lot_id,

        -- Scrap reason: raw field contains a mix of structured codes and
        -- free-text operator entries. Clean field normalizes to the controlled
        -- vocabulary. Both preserved for audit purposes.
        scrap_reason_clean                          as scrap_reason,
        scrap_reason_raw,

        defect_code_clean                           as defect_code,
        cast(quantity_scrapped as integer)          as quantity_scrapped,
        cast(quantity_reworked as integer)          as quantity_reworked,
        cast(material_cost_per_unit as double)      as material_cost_per_unit,
        cast(labor_cost_per_unit as double)         as labor_cost_per_unit,
        cast(total_scrap_cost as double)            as total_scrap_cost

    from source

)

select * from staged