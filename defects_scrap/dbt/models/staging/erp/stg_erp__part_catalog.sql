with source as (

    select * from {{ source('erp', 'part_catalog') }}

),

staged as (

    select
        part_number,
        customer,
        material_type,
        complexity,
        primary_machine,
        cast(std_labor_hrs as double)   as std_labor_hrs,
        cast(unit_price as double)      as unit_price,
        cast(requires_welding as boolean) as requires_welding

    from source

)

select * from staged