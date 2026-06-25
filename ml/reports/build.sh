#!/bin/bash
set -e
cd /workspaces/mlops_mfg/analytics/reports
python3 generate_dashboard.py
python3 generate_report.py
cd /workspaces/mlops_mfg/ml/reports
cp ../../analytics/reports/dashboard.html ../../analytics/reports/report.html .
quarto render
echo "Build complete. Starting server on http://localhost:8080 ..."
cd _site
python3 -m http.server 8080
