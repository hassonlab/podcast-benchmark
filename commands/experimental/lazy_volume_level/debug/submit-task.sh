#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./commands/experimental/lazy_volume_level/debug/submit-task.sh --model MODEL --task TASK [options] [-- EXTRA_MAIN_ARGS...]

This submitter is for debugging on the experimental lazy-volume path.

Default debug policy:
  - fold_ids = [1]
  - lag = 0

It forwards to:
  ./commands/experimental/lazy_volume_level/submit-task.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

HAS_FOLD_OVERRIDE=0
HAS_LAG_OVERRIDE=0

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fold|--fold-ids)
            HAS_FOLD_OVERRIDE=1
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --lag)
            HAS_LAG_OVERRIDE=1
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --override)
            if [[ "${2:-}" != *=* ]]; then
                echo "--override expects KEY=VALUE, got: ${2:-<missing>}"
                exit 1
            fi
            case "${2}" in
                training_params.fold_ids=*)
                    HAS_FOLD_OVERRIDE=1
                    ;;
                training_params.lag=*|training_params.min_lag=*|training_params.max_lag=*|training_params.lag_step_size=*)
                    HAS_LAG_OVERRIDE=1
                    ;;
            esac
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$HAS_FOLD_OVERRIDE" != "1" ]]; then
    ARGS+=(--fold 1)
fi

if [[ "$HAS_LAG_OVERRIDE" != "1" ]]; then
    ARGS+=(--lag 0)
fi

exec "${PROJECT_ROOT}/commands/experimental/lazy_volume_level/submit-task.sh" "${ARGS[@]}"
