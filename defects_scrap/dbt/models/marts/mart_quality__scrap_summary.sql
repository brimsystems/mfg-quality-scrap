-- mart_quality__scrap_summary
-- ---------------------------------------------------------------------------
-- Scrap event fact table with full dimensional context and derived cost
-- columns. One row per scrap event. Aggregation across dimensions is
-- handled by Power BI measures rather than pre-aggregated here, keeping
-- the mart flexible for ad-hoc slicing.
--
-- material_cost_total and labor_cost_total are derived here rather than
-- in the intermediate layer because they represent business-level cost
-- attribution logic that belongs in the mart.
--
-- Grain: one row per scrap event (inherited from int_quality__scrap_costs).
-- ---------------------------------------------------------------------------

with scrap as (

    select * from {{ ref('int_quality__scrap_costs') }}

),

final as (

    select
        -- ── Keys ──────────────────────────────────────────────────────────
        scrap_id,
        work_order_id,
        inspection_id,

        -- ── Date dimensions ───────────────────────────────────────────────
        scrap_date,
        scrap_month,
        extract('year'  from scrap_date)::integer   as scrap_year,
        extract('month' from scrap_date)::integer   as scrap_month_num,

        -- ── Dimensional context (from enriched spine) ─────────────────────
        part_number,
        customer,
        machine_id,
        machine_name,
        machine_type,
        machine_age_years,
        operator_id,
        operator_name,
        shift_code,
        complexity,
        material_type,
        supplier,
        lot_id,
        lot_cert_status,
        welding_cert_current,
        welding_cert_mismatch,

        -- ── Scrap event details ────────────────────────────────────────────
        scrap_reason,
        defect_code,
        quantity_scrapped,
        quantity_reworked,

        -- ── Cost columns ───────────────────────────────────────────────────
        material_cost_per_unit,
        labor_cost_per_unit,
        total_scrap_cost,

        -- Derived cost attribution by type
        round(
            material_cost_per_unit * quantity_scrapped, 2
        )                                           as material_cost_total,

        round(
            labor_cost_per_unit * quantity_scrapped, 2
        )                                           as labor_cost_total,

        -- ── Order-level context ────────────────────────────────────────────
        order_defect_rate,

        -- ── Pattern flags (inherited from enriched) ────────────────────────
        case
            when machine_type = 'Press Brake'
             and shift_code   = 'Shift B'
            then true else false
        end                                         as is_p1_combination,

        case
            when supplier = 'Supplier C'
            then true else false
        end                                         as is_p2_supplier,

        case
            when complexity = 'High'
            then true else false
        end                                         as is_p3_complexity,

        welding_cert_mismatch                       as is_p4_cert_mismatch

    from scrap

)

select * from final
