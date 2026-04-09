#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Active jobs for ${USER}:"
    squeue -u "${USER}" -o "%.18i %.9T %.40j %.8M %.20R"
    exit 0
fi

JOB_IDS=("$@")
JOB_IDS_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"

echo "squeue:"
squeue -j "${JOB_IDS_CSV}" -o "%.18i %.9T %.40j %.8M %.20R" || true

echo
echo "sacct:"
sacct -j "${JOB_IDS_CSV}" --format=JobID,JobName%45,State,ExitCode,Elapsed,MaxRSS -P
