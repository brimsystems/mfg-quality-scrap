with source as (

    select * from {{ source('erp', 'production_orders') }}

),

staged as (

    select
        work_order_id,

        -- Part number: normalize all format variants to canonical P-NNNN.
        -- Raw field preserved for traceability.
        part_number_clean                                       as part_number,
        part_number_raw,

        customer,
        cast(quantity_ordered as integer)                       as quantity_ordered,
        machine_id,

        -- Operator ID: raw field sometimes contains a name string rather than
        -- an OP### code. Clean field is always the canonical ID.
        operator_id_clean                                       as operator_id,
        operator_id_raw,

        -- Shift code: ~10% of records are null due to retroactive logging gaps.
        -- Where null, derive from actual_start timestamp using known shift windows:
        --   Shift A: 06:00 – 13:59
        --   Shift B: 14:00 – 21:59
        -- Records outside both windows (weekend runs, overtime) remain null.
        coalesce(
            shift_code,
            case
                when cast(actual_start as timestamp)::time
                     between '06:00:00' and '13:59:59' then 'Shift A'
                when cast(actual_start as timestamp)::time
                     between '14:00:00' and '21:59:59' then 'Shift B'
                else null
            end
        )                                                       as shift_code,

        -- Lot ID: canonical field used as join key to material_lots.
        -- Raw field preserved; ~15% of raw values are null (material not scanned
        -- at job start) — this is structural and expected, not an error.
        lot_id_clean                                            as lot_id,
        lot_id_raw,

        cast(order_date as date)                                as order_date,
        cast(scheduled_start as timestamp)                      as scheduled_start,
        cast(actual_start as timestamp)                         as actual_start,
        complexity,
        material_type,
        cast(requires_welding as boolean)                       as requires_welding,
        cast(std_labor_hrs as double)                           as std_labor_hrs

    from source

)

select * from staged