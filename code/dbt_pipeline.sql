-- dbt transformation layer (DuckDB): staging, intermediate and marts. Sources are read from generated CSV extracts.

-- ########################################################################
-- STAGING LAYER
-- ########################################################################

-- ====================================================================
-- model: stg_erp__part_catalog
-- ====================================================================

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

-- ====================================================================
-- model: stg_erp__production_orders
-- ====================================================================

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

-- ====================================================================
-- model: stg_hr__operators
-- ====================================================================

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

-- ====================================================================
-- model: stg_materials__lots
-- ====================================================================

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

-- ====================================================================
-- model: stg_mes__machines
-- ====================================================================

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

-- ====================================================================
-- model: stg_qms__inspection_records
-- ====================================================================

with source as (

    select * from {{ source('qms', 'inspection_records') }}

),

staged as (

    select
        inspection_id,
        work_order_id,
        cast(inspection_date as timestamp)      as inspection_date,
        inspector_id,
        cast(quantity_inspected as integer)     as quantity_inspected,
        cast(quantity_passed as integer)        as quantity_passed,
        cast(quantity_failed as integer)        as quantity_failed,

        -- Defect code: raw field contains free-text variants and typos.
        -- Clean field contains the canonical value from the controlled vocabulary.
        -- Both preserved — raw for audit, clean for analysis.
        defect_code_clean                       as defect_code,
        defect_code_raw,

        disposition,
        notes,

        -- Flag records with timestamps outside shift windows.
        -- These are retroactive entries and should not be used for
        -- time-of-day analysis without review.
        case
            when cast(inspection_date as timestamp)::time
                 not between '06:00:00' and '21:59:59'
            then true
            else false
        end                                     as is_anomalous_timestamp

    from source

),

deduplicated as (

    -- The QMS creates duplicate inspection records when a session times out
    -- during entry and the inspector re-submits. Duplicates share work_order_id
    -- but have distinct inspection_ids. The earliest record is retained as the
    -- authoritative entry; subsequent records are discarded.
    select *
    from staged
    qualify row_number() over (
        partition by work_order_id
        order by inspection_id asc
    ) = 1

)

select * from deduplicated

-- ====================================================================
-- model: stg_qms__scrap_events
-- ====================================================================

with source as (

    select * from {{ source('qms', 'scrap_events') }}

),

staged as (

    select
        scrap_id,
        work_order_id,
        inspection_id,
        cast(scrap_date as timestamp)               as scrap_date,
        machine_id,
        operator_id,
        shift_code,
        material_type,
        lot_id,

        -- Scrap reason: raw field contains a mix of structured codes and
        -- free-text operator entries. Clean field normalizes to the controlled
        -- vocabulary. Both preserved for audit purposes.
        scrap_reason_clean                          as scrap_reason,
        scrap_reason_raw,

        defect_code_clean                           as defect_code,
        cast(quantity_scrapped as integer)          as quantity_scrapped,
        cast(quantity_reworked as integer)          as quantity_reworked,
        cast(material_cost_per_unit as double)      as material_cost_per_unit,
        cast(labor_cost_per_unit as double)         as labor_cost_per_unit,
        cast(total_scrap_cost as double)            as total_scrap_cost

    from source

)

select * from staged

-- ########################################################################
-- INTERMEDIATE LAYER
-- ########################################################################

-- ====================================================================
-- model: int_quality__orders_enriched
-- ====================================================================

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

-- ====================================================================
-- model: int_quality__orders_with_inspections
-- ====================================================================

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

-- ====================================================================
-- model: int_quality__scrap_costs
-- ====================================================================

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

-- ########################################################################
-- MARTS LAYER
-- ########################################################################

-- ====================================================================
-- model: mart_quality__defect_rates
-- ====================================================================

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

        case when quantity_failed > 0 then true else false end  as defect_flag,

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

-- ====================================================================
-- model: mart_quality__operator_performance
-- ====================================================================

-- mart_quality__operator_performance
-- ---------------------------------------------------------------------------
-- Operator-level performance summary. One row per operator. Pre-aggregated
-- because the operator scorecard is the primary use case and the aggregation
-- logic is stable — defect rates, job counts, and cost attribution are
-- unlikely to require re-slicing at a lower grain in Power BI.
--
-- peer_mean_defect_rate and defect_rate_vs_peer enable direct comparison
-- of each operator against the fleet average without requiring a DAX measure.
--
-- Grain: one row per operator.
-- ---------------------------------------------------------------------------

with defect_rates as (

    select * from {{ ref('mart_quality__defect_rates') }}

),

scrap_by_operator as (

    select
        operator_id,
        sum(total_scrap_cost)   as total_scrap_cost_attributed

    from {{ ref('mart_quality__scrap_summary') }}
    group by operator_id

),

operator_stats as (

    select
        operator_id,
        operator_name,
        cert_level,
        specialization,
        welding_cert_current,
        hire_date,

        count(*)                                            as total_jobs,

        sum(case when requires_welding then 1 else 0 end)  as total_welding_jobs,

        sum(case when welding_cert_mismatch then 1 else 0 end)
                                                            as cert_mismatch_job_count,

        round(avg(defect_rate), 4)                         as mean_defect_rate,

        round(
            percentile_cont(0.5) within group (
                order by defect_rate
            ), 4
        )                                                   as median_defect_rate,

        sum(quantity_failed)                                as total_quantity_failed,

        sum(case when defect_flag then 1 else 0 end)         as jobs_with_defect,

        round(
            sum(case when defect_flag then 1 else 0 end)::double
            / nullif(count(*), 0) * 100, 1
        )                                                   as pct_jobs_with_defect

    from defect_rates
    where defect_rate is not null
    group by
        operator_id, operator_name, cert_level, specialization,
        welding_cert_current, hire_date

),

fleet_mean as (

    select round(avg(defect_rate), 4) as peer_mean_defect_rate
    from defect_rates
    where defect_rate is not null

),

final as (

    select
        o.operator_id,
        o.operator_name,
        o.cert_level,
        o.specialization,
        o.welding_cert_current,
        o.hire_date,
        o.total_jobs,
        o.total_welding_jobs,
        o.cert_mismatch_job_count,
        o.mean_defect_rate,
        o.median_defect_rate,
        o.total_quantity_failed,
        o.jobs_with_defect,
        o.pct_jobs_with_defect,
        f.peer_mean_defect_rate,
        round(
            o.mean_defect_rate - f.peer_mean_defect_rate, 4
        )                                                   as defect_rate_vs_peer,
        coalesce(s.total_scrap_cost_attributed, 0)         as total_scrap_cost_attributed

    from operator_stats o
    cross join fleet_mean f
    left join scrap_by_operator s
        on o.operator_id = s.operator_id

)

select * from final

-- ====================================================================
-- model: mart_quality__scrap_summary
-- ====================================================================

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
