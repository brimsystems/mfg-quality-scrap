with source as (

    select * from {{ source('hr', 'operators') }}

),

staged as (

    select
        operator_id,
        operator_name,
        shift,
        cast(hire_date as date)     as hire_date,
        cert_level,
        specialization,
        welding_cert_current

    from source

)

select * from staged