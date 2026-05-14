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

        sum(case when is_defect then 1 else 0 end)         as jobs_with_defect,

        round(
            sum(case when is_defect then 1 else 0 end)::double
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
