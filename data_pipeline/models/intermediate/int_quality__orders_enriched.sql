-- int_quality__orders_enriched
-- ---------------------------------------------------------------------------
-- Extends the orders-with-inspections spine with dimensional context from
-- three additional source systems: MES (machine attributes), HR (operator
-- certification status), and Materials/WMS (supplier and lot certification).
--
-- This is the first model in the pipeline where all four hidden patterns
-- are simultaneously visible:
--
--   P1 — Shift B × machine_type='Press Brake' × age_years (MES join)
--   P2 — supplier × cert_status (Materials join)
--   P3 — complexity (already on the spine from ERP part catalog)
--   P4 — operator_id × welding_cert_current=false (HR join)
--
-- Grain: one row per work order (inherited from the spine).
--
-- Join notes:
--   - machines: INNER JOIN — every production order must reference a valid
--     machine. Missing machine records indicate upstream ERP data entry gaps
--     and should be investigated rather than silently dropped.
--   - operators: INNER JOIN — same rationale as machines.
--   - material_lots: LEFT JOIN — lot_id is ~15% null in production_orders
--     (material not scanned at job start). These rows are retained with
--     NULL supplier/cert_status dimensions.
-- ---------------------------------------------------------------------------

with spine as (

    select * from {{ ref('int_quality__orders_with_inspections') }}

),

machines as (

    select
        machine_id,
        machine_name,
        machine_type,
        age_years,
        location        as machine_location

    from {{ ref('stg_mes__machines') }}

),

operators as (

    select
        operator_id,
        operator_name,
        shift           as operator_home_shift,
        cert_level,
        specialization,
        welding_cert_current,
        hire_date

    from {{ ref('stg_hr__operators') }}

),

lots as (

    select
        lot_id,
        supplier,
        cert_status     as lot_cert_status,
        receipt_date    as lot_receipt_date,
        unit_cost_per_lb

    from {{ ref('stg_materials__lots') }}

),

enriched as (

    select
        -- ── Keys ──────────────────────────────────────────────────────────
        s.work_order_id,
        s.inspection_id,

        -- ── Order dimensions (from spine) ──────────────────────────────────
        s.part_number,
        s.customer,
        s.shift_code,
        s.complexity,
        s.material_type,
        s.requires_welding,
        s.lot_id,

        -- ── Dates ─────────────────────────────────────────────────────────
        s.order_date,
        s.actual_start,
        s.inspection_date,

        -- ── Inspection metrics (from spine) ────────────────────────────────
        s.quantity_ordered,
        s.quantity_inspected,
        s.quantity_passed,
        s.quantity_failed,
        s.defect_rate,
        s.defect_code,
        s.disposition,
        s.is_anomalous_timestamp,
        s.schedule_variance_hrs,
        s.std_labor_hrs,
        s.inspector_id,

        -- ── Machine context (MES) ──────────────────────────────────────────
        s.machine_id,
        m.machine_name,
        m.machine_type,
        m.age_years                 as machine_age_years,
        m.machine_location,

        -- ── Operator context (HR) ──────────────────────────────────────────
        s.operator_id,
        o.operator_name,
        o.operator_home_shift,
        o.cert_level,
        o.specialization,
        o.welding_cert_current,
        o.hire_date,

        -- Flag: operator assigned to a welding job without a current cert.
        -- This is the direct indicator for Pattern 4.
        case
            when s.requires_welding = true
             and o.welding_cert_current = false
            then true
            else false
        end                         as welding_cert_mismatch,

        -- ── Material/supplier context (WMS) ────────────────────────────────
        l.supplier,
        l.lot_cert_status,
        l.lot_receipt_date,
        l.unit_cost_per_lb

    from spine s
    inner join machines m
        on s.machine_id = m.machine_id
    inner join operators o
        on s.operator_id = o.operator_id
    left join lots l
        on s.lot_id = l.lot_id

)

select * from enriched
