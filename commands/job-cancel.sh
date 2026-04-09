#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./commands/job-cancel.sh JOB_ID [JOB_ID...]"
    exit 1
fi

scancel "$@"
echo "Cancelled: $*"
