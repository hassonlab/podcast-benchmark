#!/usr/bin/env bash
set -euo pipefail

MIN_LAG="${MIN_LAG:--1000}"
MAX_LAG="${MAX_LAG:-1000}"
LAG_STEP="${LAG_STEP:-100}"
LAGS_PER_JOB="${LAGS_PER_JOB:-2}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_CONFIGS=(
  "configs/foundation_models/diver/volume_level/persubject_concat.yml"
  "configs/foundation_models/diver/volume_level/subject1_full.yml"
  "configs/foundation_models/diver/volume_level/subject2_full.yml"
  "configs/foundation_models/diver/volume_level/subject3_full.yml"
  "configs/foundation_models/diver/volume_level/subject4_full.yml"
  "configs/foundation_models/diver/volume_level/subject5_full.yml"
  "configs/foundation_models/diver/volume_level/subject6_full.yml"
  "configs/foundation_models/diver/volume_level/subject7_full.yml"
  "configs/foundation_models/diver/volume_level/subject8_full.yml"
  "configs/foundation_models/diver/volume_level/subject9_full.yml"
)

if [[ -n "${CONFIGS:-}" ]]; then
  read -r -a CONFIG_LIST <<< "$CONFIGS"
else
  CONFIG_LIST=("${DEFAULT_CONFIGS[@]}")
fi

if (( LAG_STEP <= 0 )); then
  echo "LAG_STEP must be greater than 0" >&2
  exit 1
fi

if (( LAGS_PER_JOB <= 0 )); then
  echo "LAGS_PER_JOB must be greater than 0" >&2
  exit 1
fi

if (( MIN_LAG > MAX_LAG )); then
  echo "MIN_LAG must be less than or equal to MAX_LAG" >&2
  exit 1
fi

mkdir -p logs

submitted=0

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

config_job_name() {
  local config="$1"
  local name
  name="$(basename "$config")"
  name="${name%.yml}"
  name="${name%.yaml}"
  name="${name//_/-}"
  printf 'decoder-training-diver-volume-%s\n' "$name"
}

submit_batch() {
  local config="$1"
  local batch_min="$2"
  local batch_max="$3"
  local job_name="$4"
  local cmd=(
    sbatch
    --job-name="$job_name"
    --dependency=singleton
  )
  local sbatch_flags=()
  local config_overrides=()

  if [[ -n "${SBATCH_FLAGS:-}" ]]; then
    read -r -a sbatch_flags <<< "$SBATCH_FLAGS"
    cmd+=("${sbatch_flags[@]}")
  fi

  cmd+=(
    submit.sh
    main.py
    --config "$config"
    --training_params.min_lag="$batch_min"
    --training_params.max_lag="$batch_max"
    --training_params.lag_step_size="$LAG_STEP"
  )

  if [[ -n "${CONFIG_OVERRIDES:-}" ]]; then
    read -r -a config_overrides <<< "$CONFIG_OVERRIDES"
    cmd+=("${config_overrides[@]}")
  fi

  echo "Submitting ${config}: ${batch_min}..${batch_max} ms as ${job_name}"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
}

for config in "${CONFIG_LIST[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Config not found: $config" >&2
    exit 1
  fi

  job_name="$(config_job_name "$config")"
  lag="$MIN_LAG"
  while (( lag <= MAX_LAG )); do
    batch_max=$((lag + (LAGS_PER_JOB - 1) * LAG_STEP))
    if (( batch_max > MAX_LAG )); then
      batch_max="$MAX_LAG"
    fi
    submit_batch "$config" "$lag" "$batch_max" "$job_name"
    lag=$((batch_max + LAG_STEP))
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete: ${submitted} jobs would be submitted."
else
  echo "Submitted ${submitted} jobs."
fi
