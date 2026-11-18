#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

PBS_SCRIPT="${PBS_SCRIPT:-pbs/target_full_train.pbs}"
if [[ $# -gt 0 ]]; then
  export TARGET_TYPES="$1"
fi
export TARGET_TYPES="${TARGET_TYPES:-metabolite,protein,rna,gene}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub not found. Run this on the PBS login node, or execute the PBS script with bash for local smoke testing." >&2
  exit 127
fi

echo "Submitting $PBS_SCRIPT with TARGET_TYPES=$TARGET_TYPES"
qsub -V "$PBS_SCRIPT"
