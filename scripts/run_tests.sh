#!/usr/bin/env bash
# Proje kökünden testleri çalıştırır. Önce: pip install -r requirements.txt
set -e
cd "$(dirname "$0")/.."
python -m pytest tests/ -v --tb=short
