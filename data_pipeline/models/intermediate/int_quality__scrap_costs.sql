-- int_quality__scrap_costs
-- ---------------------------------------------------------------------------
-- Joins scrap events to the enriched order spine, attaching the full
-- dimensional context to each scrap cost record.
--
-- Grain: one row per scrap event (from stg_qms__scrap_events). A single
-- work order may have multiple scrap events if defects occurred at different
-- process steps or were logged across multiple sessions.
--
-- Join note:
--   The join uses work_order_id, not inspection_id. This is intentional —
--   approximately 606 scrap events reference inspection_ids that no longer
--   exist in stg_qms__inspection_records (dropped during deduplication).
--   Joining on work_order_id avoids losing those cost records.
--
--   Scrap events with no matching enriched order (i.e., the work order had
--   no inspection record) are excluded via INNER JOIN. These are rare edge
--   cases where scrap was logged against a work order that was never formally
--   inspected — they do not carry a usable defect rate and cannot be
--   attributed to the analytical dimensions required for Pattern analysis.
-- ---------------------------------------------------------------------------

with scrap as (

    select
        scrap_id,
        work_order_id,
        inspection_id,
        scrap_date,
        scrap_reason,
        defect_code,
        quantity_scrapped,
        quantity_reworked,
        material_cost_per_unit,
        labor_cost_per_unit,
        total_scrap_cost

    from {{ ref('stg_qms__scrap_events') }}

),

enriched as (

    select
        work_order_id,
        inspection_id,
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
        defect_rate,
        order_date,
        actual_start

    from {{ ref('int_quality__orders_enriched') }}

),

joined as (

    select
        -- ── Keys ──────────────────────────────────────────────────────────
        sc.scrap_id,
        sc.work_order_id,
        sc.inspection_id,

        -- ── Scrap event details ────────────────────────────────────────────
        sc.scrap_date,
        sc.scrap_reason,
        sc.defect_code,
        sc.quantity_scrapped,
        sc.quantity_reworked,
        sc.material_cost_per_unit,
        sc.labor_cost_per_unit,
        sc.total_scrap_cost,

        -- ── Order and inspection context ───────────────────────────────────
        e.part_number,
        e.customer,
        e.machine_id,
        e.machine_name,
        e.machine_type,
        e.machine_age_years,
        e.operator_id,
        e.operator_name,
        e.shift_code,
        e.complexity,
        e.material_type,
        e.supplier,
        e.lot_id,
        e.lot_cert_status,
        e.welding_cert_current,
        e.welding_cert_mismatch,

        -- Order-level defect rate for cross-referencing magnitude vs cost
        e.defect_rate                   as order_defect_rate,

        -- Date dimensions for trend analysis
        e.order_date,
        e.actual_start,
        sc.scrap_date,
        date_trunc('month', sc.scrap_date)  as scrap_month

    from scrap sc
    inner join enriched e
        on sc.work_order_id = e.work_order_id

)

select * from joined
