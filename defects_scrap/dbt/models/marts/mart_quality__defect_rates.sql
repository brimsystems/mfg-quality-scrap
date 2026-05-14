-- mart_quality__defect_rates
-- ---------------------------------------------------------------------------
-- Primary analytical table for quality performance reporting and ML modeling.
-- One row per work order with all dimensional context and quality metrics
-- attached. Serves as the main fact table for Power BI and the feature table
-- for defect risk classification.
--
-- Pattern flags provide pre-computed boolean indicators for each of the four
-- cross-system patterns, making dashboard filtering straightforward without
-- requiring end users to know specific machine IDs or operator codes.
--
-- Grain: one row per work order (inherited from int_quality__orders_enriched).
-- ---------------------------------------------------------------------------

with enriched as (

    select * from {{ ref('int_quality__orders_enriched') }}

),

final as (

    select
        -- ── Keys ──────────────────────────────────────────────────────────
        work_order_id,
        inspection_id,
        part_number,
        customer,

        -- ── Date dimensions ───────────────────────────────────────────────
        order_date,
        actual_start,
        inspection_date,
        date_trunc('month', actual_start)           as order_month,
        extract('year'  from actual_start)::integer as order_year,
        extract('month' from actual_start)::integer as order_month_num,

        -- ── Machine dimensions (MES) ───────────────────────────────────────
        machine_id,
        machine_name,
        machine_type,
        machine_age_years,
        machine_location,

        -- ── Operator dimensions (HR) ───────────────────────────────────────
        operator_id,
        operator_name,
        operator_home_shift,
        cert_level,
        specialization,
        welding_cert_current,
        hire_date,

        -- ── Order dimensions (ERP) ─────────────────────────────────────────
        shift_code,
        complexity,
        material_type,
        requires_welding,
        std_labor_hrs,
        schedule_variance_hrs,
        quantity_ordered,

        -- ── Material/supplier dimensions (WMS) ────────────────────────────
        lot_id,
        supplier,
        lot_cert_status,
        lot_receipt_date,
        unit_cost_per_lb,

        -- ── Inspection metrics (QMS) ───────────────────────────────────────
        quantity_inspected,
        quantity_passed,
        quantity_failed,
        defect_rate,
        defect_code,
        disposition,
        is_anomalous_timestamp,
        inspector_id,

        -- ── Derived quality flags ──────────────────────────────────────────
        welding_cert_mismatch,

        case when quantity_failed > 0 then true else false end  as is_defect,

        -- ── Pattern flags ──────────────────────────────────────────────────
        -- P1: Press Brake jobs on Shift B — elevated defect rate (3.4x)
        case
            when machine_type = 'Press Brake'
             and shift_code   = 'Shift B'
            then true else false
        end                                         as is_p1_combination,

        -- P2: Supplier C material — elevated defect rate (1.9x)
        case
            when supplier = 'Supplier C'
            then true else false
        end                                         as is_p2_supplier,

        -- P3: High-complexity parts — elevated defect rate (1.6x)
        case
            when complexity = 'High'
            then true else false
        end                                         as is_p3_complexity,

        -- P4: Welding job assigned to operator with lapsed certification (2.2x)
        welding_cert_mismatch                       as is_p4_cert_mismatch

    from enriched

)

select * from final
