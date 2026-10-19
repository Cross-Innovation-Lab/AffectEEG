#!/usr/bin/env bash
# Sequential main-experiment matrix. Prepare data before launching this script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
AFFECTEEG_PYTHON="${AFFECTEEG_PYTHON:-python}"
AFFECTEEG_DEVICE="${AFFECTEEG_DEVICE:-auto}"
AFFECTEEG_PROCESSED="${AFFECTEEG_PROCESSED:-data/processed}"
AFFECTEEG_OUTPUT="${AFFECTEEG_OUTPUT:-outputs}"
runtime=(--processed-root "$AFFECTEEG_PROCESSED" --output-root "$AFFECTEEG_OUTPUT" --device "$AFFECTEEG_DEVICE")
"$AFFECTEEG_PYTHON" -m affecteeg train-affect "${runtime[@]}"
"$AFFECTEEG_PYTHON" -m affecteeg train-tone "${runtime[@]}"
"$AFFECTEEG_PYTHON" -m affecteeg evaluate --task tone "${runtime[@]}"
"$AFFECTEEG_PYTHON" -m affecteeg train-reconstruction "${runtime[@]}"
"$AFFECTEEG_PYTHON" -m affecteeg evaluate --task reconstruction "${runtime[@]}"
