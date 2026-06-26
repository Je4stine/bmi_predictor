#!/usr/bin/env bash
# Wrapper that runs data_curation.py in fresh subprocesses for each chunk, so
# any cumulative MediaPipe / Metal state is reset between chunks.

set -u
DATA_DIR=${1:-visual_bmi}
CHUNK=${CHUNK:-50}  # keep small; MediaPipe slows after ~60 calls/process on macOS
CSV="$DATA_DIR/curation.csv"
FILTER='clearcut|Source Location Trace|wireless/android'

prev=0
while true; do
    GLOG_minloglevel=2 python3 -u data_curation.py \
        --data_dir "$DATA_DIR" \
        --limit "$CHUNK" \
        --extract_only \
        2>&1 | grep -v -E "$FILTER"

    if [ ! -f "$CSV" ]; then
        echo "ERROR: $CSV missing after run" >&2
        exit 1
    fi
    cur=$(($(wc -l < "$CSV") - 1))  # minus header
    echo "=== chunk done: csv has $cur rows (was $prev) ==="
    if [ "$cur" -le "$prev" ]; then
        echo "=== no progress this chunk, exiting extraction loop ==="
        break
    fi
    prev=$cur
done

echo "=== running final clustering + preview pass ==="
GLOG_minloglevel=2 python3 -u data_curation.py \
    --data_dir "$DATA_DIR" \
    --k 5 --preview \
    2>&1 | grep -v -E "$FILTER"
