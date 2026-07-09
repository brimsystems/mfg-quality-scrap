# Manufacturing Data Platform & Defect Risk Intelligence 

**An end-to-end, simulated data platform for a metal fabrication shop: from raw multi-system data to analytics (diagnostic report and KPI dashboard) and a machine learning (ML) model that flags pre-production defect risks, embedded inside the existing ERP system.**

> Built by Brian Davis — fractional data engineering & analytics partner for SMB manufacturers.

---

## Situation Overview

A sheet-metal fabricator (~$30M revenue, two shifts, four machine groups) was losing margin to elevated defect rates, but couldn't see *why*. The company was already capturing the data needed to understand the drivers of defects. However, the data was stored across disconnected systems: machine health and production data was stored in the MES; supplier, operator, and schedule data was stored in the ERP; and inspection outcomes were stored in the QMS. These data silos meant that the combinations of operating conditions that actually drive defects, for example, an aging machine running a high-complexity job on a thin-gauge lot late in the schedule, went unseen until defects already occured.

This project addresses that situation end-to-end: it integrates the three data systems, diagnoses where the cost actually concentrates, builds a machine learning model that scores each work order's defect risk before it runs, delivers that score within the existing work-order system so that operators can see it live, then monitors that model's effectiveness over time using MLOps best practices.

---

## Deliverables

### 1. ERP work-order queue with embedded defect risk flags — *primary deliverable*
A simulated shop-floor work-order queue, with each job's defect-risk tier and top risk driver surfaced inline. By embedding the model within a JobBOSS-style ERP system, operators can see within systems they already use which jobs need a second look, and why, in plain operational language.

> **[Open the live ERP work-order queue with embedded defect risk flags →](https://brimsystems.github.io/mfg-dataplatform-defectrisk/)**

### 2. Analytics diagnostic report — *understanding the drivers of defects and scrap costs*

An adhoc quality and scrap diagnostic: where defects and scrap costs concentrate by machine, shift, operator, material, supplier, and complexity, and the cross-system combinations that compound risk of defects occuring. 

> **[Open the analytics diagnostic report →](https://brimsystems.github.io/mfg-dataplatform-defectrisk/reports/report.html)**

### 3. Analytics dashboard — *monthly view of defect rate and scrap cost KPIs*

Drivers of defect risks and scrap costs with historical trends. Represents the recurring operational view that managers would use to understand their business.

> **[Open the analytics dashboard →](https://brimsystems.github.io/mfg-dataplatform-defectrisk/reports/dashboard.html)**

### 4. ML model overview — *plain-language model summary*

A concise model card for a non-technical stakeholder: what the ML model predicts, what it was trained on, how it performs at a glance, and where its limits are. Answers the question, "what does this model do and can I trust it?"

> **[Open the ML model overview →](https://brimsystems.github.io/mfg-dataplatform-defectrisk/reports/ml_overview.html)**

### 5. ML model technical report — *depth for the technical evaluator*

The full technical detail: training data summary, model selection comparison, validation/test performance, calibration, confusion matrix, and SHAP-based feature importance. For the reader who wants to verify the rigor underneath the outputs above.

> **[Open the ML model technical report →](https://brimsystems.github.io/mfg-dataplatform-defectrisk/reports/ml_technical.html)**

### 6. MLOps monitoring report — *how the model performs over time in production*

A production-monitoring dashboard tracking the model across periods on four layers (performance, target drift, prediction drift, and feature drift), with an explicit, rules-based retraining decision. This report reflects MLOps best practices and answers "is this model producing reliable results over time that we can continue to rely on?"

> **[Open the MLOps monitoring report →](https://brimsystems.github.io/mfg-dataplatform-defectrisk/reports/monitoring_report.html)**

---

## How it works

```
Data source systems  →  dbt integration & pipeline  →  data marts  →  analytics (diagnosis, dashboard & ML) → MLOps monitoring

```

This project works end-to-end, moving left to right in the sequence above. Raw extracts from three data source systems with the integration problems that come with them, including mismatched IDs, inconsistent naming, and varying granularity, are combined using a tested dbt pipeline into data marts. Those marts feed the analytics report and dashboard as well as the machine learning pipeline. The data is split into training, validation and test sets, and the machine learning model is then built. Ongoing model scoring is set up using MLOps best practices and runs as a monthly batch, with each period monitored against training and validation references.

---

## Repository structure

```
metalfab-data-platform/
├── README.md
├── pyproject.toml · poetry.lock · LICENSE · .gitignore
├── data-source/               ←   data generation + the DuckDB warehouse
│   └── generate/
├── data-pipeline/             ←   dbt project: staging → marts, with tests
│   └── models/
├── analytics/                 ←   diagnostic report + operating dashboard
│   └── reports/               ←   generator scripts and working outputs
├── ml/                        ←   the machine learning model lifecycle
│   ├── src/
│   │   ├── features.py        ←   shared feature engineering (single source of truth)
│   │   ├── training.py        ←   train, select, register; emits validation reference
│   │   ├── scoring.py         ←   monthly batch scoring + SHAP drivers
│   │   └── monitoring.py      ←   four-layer drift + performance monitoring
│   ├── notebooks/             ←   analytical prep (read-only documentation)
│   └── reports/               ←   ML report generators (overview, technical, monitoring, ERP)
└── docs/                      ←   what GitHub Pages serves
    ├── index.html             ←   the ERP demo (root URL)
    ├── reports/               ←   rendered HTML reports
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Integration & transformation | dbt, DuckDB (local dev) / Azure SQL (production target) |
| Analytics & reporting | Python, pandas, matplotlib, HTML/CSS |
| Modeling | XGBoost, scikit-learn, Optuna, SHAP |
| MLOps | MLflow (tracking & registry), Evidently (drift), Prefect (orchestration) |
| Delivery | Static HTML/JS (ERP simulation), GitHub Pages |

Stack reflects personal expertise and project constraints; real engagements are tool-agnostic and meet the client where their systems already are.

---

## Reproduce locally

<details>
<summary>Setup and run order</summary>

```bash
# 1. Environment 
python3 -m venv .venv
source .venv/bin/activate
pip install -e .          # installs the project + dependencies declared in pyproject.toml

# 2. Generate data and build the warehouse
python3 -m data_source.generate.run_generator     # adjust to your generator entry point
cd data-pipeline && dbt build && cd ..

# 3. Analytics
cd analytics/reports && python3 generate_report.py && python3 generate_dashboard.py && cd ../..

# 4. ML lifecycle (train → score → monitor)
cd ml
python3 src/training.py        # trains, registers Production model, writes validation reference
python3 src/scoring.py         # monthly batch scoring (Jan–Mar)
python3 src/monitoring.py      # four-layer monitoring across periods
cd ..

# 5. Client deliverables
cd ml/reports && python3 generate_erp_dashboard.py && python3 generate_monitoring_report.py && cd ../..
```

</details>

---

## A note on the data

This project runs on a representative dataset modeled on a metal fabrication shop. It was generated to reflect the real integration challenges and operational patterns of the sector, without using any client's data. That's deliberate: it demonstrates the full diagnosis-to-execution workflow end to end, on data I can share publicly. The analytics and ML model results are therefore illustrative, *and do not reflect any specific shop*.

---

## About

Brian Davis is a fractional data engineering and analytics partner for small and mid-sized  manufacturers. Through embedded partnership rather than transactional consulting, I implement data-driven operational improvements that create immediate and lasting business value. 

**[Brian Davis]** — brian@brimsystems.com 

