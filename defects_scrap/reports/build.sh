#!/bin/bash
set -e
cd /workspaces/portfolio/defects_scrap/reports
python3 generate_dashboard.py
python3 generate_report.py
quarto render
echo "Build complete. Starting server on http://localhost:8080 ..."
cd _site
python3 -m http.server 8080
