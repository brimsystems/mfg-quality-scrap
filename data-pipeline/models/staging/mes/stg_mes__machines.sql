with source as (

    select * from {{ source('mes', 'machines') }}

),

staged as (

    select
        machine_id,
        machine_name,
        machine_type,
        cast(age_years as integer)  as age_years,
        location

    from source

)

select * from staged