#!/bin/bash
# Runs pipeline steps 1-8. Steps 9-10 need your own photos first.
#
# Python resolution order (override with PY=/path/to/python ./run_pipeline.sh):
#   1. $PY if set
#   2. ~/.venvs/sinbmi312 — local venv OUTSIDE iCloud (preferred: the iCloud
#      venv312 gets evicted under disk pressure and then TF import stalls
#      while files re-download one read at a time)
#   3. ../venv312 — the original in-repo venv
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PY:-}" ]; then
  if [ -x "$HOME/.venvs/sinbmi312/bin/python" ]; then
    PY="$HOME/.venvs/sinbmi312/bin/python"
  else
    PY=../venv312/bin/python
  fi
fi
echo "Using python: $PY"

$PY step1_clean_visualbmi.py
$PY step2_clean_image2bmi.py
$PY step3_make_common_csv.py
$PY step4_preprocess.py
$PY step5_split.py
$PY step6_train_visualbmi.py 2>&1 | tee step6_train.log
$PY step7_train_combined.py 2>&1 | tee step7_train.log
$PY step8_evaluate.py
