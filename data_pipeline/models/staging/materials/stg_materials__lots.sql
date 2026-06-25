with source as (

    select * from {{ source('materials', 'material_lots') }}

),

staged as (

    select
        -- Canonical lot ID is the join key used downstream.
        -- The raw field (lot_id_raw) is preserved for audit purposes.
        lot_id_clean                        as lot_id,
        lot_id_raw,
        supplier,
        material_type,
        cast(receipt_date as date)          as receipt_date,
        cert_status,
        cast(quantity_lbs as integer)       as quantity_lbs,
        cast(unit_cost_per_lb as double)    as unit_cost_per_lb

    from source

)

select * from staged