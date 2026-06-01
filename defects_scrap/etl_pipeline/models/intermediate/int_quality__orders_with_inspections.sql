-- int_quality__orders_with_inspections
-- ---------------------------------------------------------------------------
-- Joins production work orders to their inspection outcomes. This is the
-- central fact spine for all quality analytics — every defect metric and
-- rate calculation derives from this join.
--
-- Grain: one row per work order. Work orders without an inspection record
-- are excluded (INNER JOIN) — uninspected orders carry no quality signal.
--
-- Defect rate is computed here as the primary analytical measure:
--   defect_rate = quantity_failed / quantity_inspected
-- A NULL-safe division is applied; zero-inspected rows return NULL rather
-- than dividing by zero.
-- ---------------------------------------------------------------------------

with orders as (

    select
        work_order_id,
        part_number,
        customer,
        machine_id,
        operator_id,
        shift_code,
        lot_id,
        complexity,
        material_type,
        requires_welding,
        quantity_ordered,
        std_labor_hrs,
        order_date,
        scheduled_start,
        actual_start

    from {{ ref('stg_erp__production_orders') }}

),

inspections as (

    select
        inspection_id,
        work_order_id,
        inspection_date,
        inspector_id,
        quantity_inspected,
        quantity_passed,
        quantity_failed,
        defect_code,
        disposition,
        is_anomalous_timestamp

    from {{ ref('stg_qms__inspection_records') }}

),

joined as (

    select
        -- Keys
        o.work_order_id,
        i.inspection_id,

        -- Order dimensions
        o.part_number,
        o.customer,
        o.machine_id,
        o.operator_id,
        o.shift_code,
        o.lot_id,
        o.complexity,
        o.material_type,
        o.requires_welding,

        -- Dates
        o.order_date,
        o.actual_start,
        i.inspection_date,

        -- Inspection metrics
        o.quantity_ordered,
        i.quantity_inspected,
        i.quantity_passed,
        i.quantity_failed,

        -- Defect rate (NULL where quantity_inspected = 0)
        case
            when i.quantity_inspected = 0 then null
            else cast(i.quantity_failed as double) / i.quantity_inspected
        end                             as defect_rate,

        -- Quality outcome dimensions
        i.defect_code,
        i.disposition,
        i.is_anomalous_timestamp,

        -- Scheduling variance (positive = started late)
        case
            when o.scheduled_start is not null and o.actual_start is not null
            then datediff('hour', o.scheduled_start, o.actual_start)
            else null
        end                             as schedule_variance_hrs,

        o.std_labor_hrs,
        i.inspector_id

    from orders o
    inner join inspections i
        on o.work_order_id = i.work_order_id

)

select * from joined
