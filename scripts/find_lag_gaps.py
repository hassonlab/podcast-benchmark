#!/usr/bin/env python
"""Find missing lag-result rows for configured paper-result directories."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


DEFAULT_CONFIG = Path("configs/paper_results_dense.yml")
DEFAULT_MODEL = "baseline"
DEFAULT_MIN_LAG = -800
DEFAULT_MAX_LAG = 1000
DEFAULT_STEP = 25

TASK_NAMES = {
    "content_noncontent": "content_noncontent_task",
    "gpt_surprise": "gpt_surprise_task",
    "gpt_surprise_multiclass": "gpt_surprise_multiclass_task",
    "iu_boundary": "iu_boundary_task",
    "llm_decoding": "llm_decoding_task",
    "pos": "pos_task",
    "sentence_onset": "sentence_onset_task",
    "volume_level": "volume_level_decoding_task",
    "whisper_embedding": "whisper_embedding_decoding_task",
    "word_embedding": "word_embedding_decoding_task",
}

SCOPE_TO_RUN_MODE = {
    "per_subject": "per_subject",
    "per_region": "per_region",
    "super_subject": "supersubject",
}


@dataclass(frozen=True)
class Gap:
    task: str
    scope: str
    entity: str
    missing: tuple[int, ...]
    present_count: int


@dataclass(frozen=True)
class CandidateConfig:
    path: Path
    task_name: str
    run_mode: str
    trial_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a paper-results config, scan configured lag_performance.csv files, "
            "and report missing lags plus suggested make commands."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-lag", type=int, default=DEFAULT_MIN_LAG)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument(
        "--ignore-lag",
        type=int,
        action="append",
        default=[],
        help="Lag to ignore when reporting gaps. Repeatable, e.g. --ignore-lag 1000.",
    )
    parser.add_argument(
        "--baseline-config-root",
        type=Path,
        default=Path("configs/baselines"),
        help="Root used to find source configs for suggested make commands.",
    )
    parser.add_argument(
        "--no-commands",
        action="store_true",
        help="Only print the gap report.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def resolve_path(path: str | Path, root: Path) -> Path:
    result = Path(str(path).rstrip("/"))
    if not result.is_absolute():
        result = root / result
    return result


def read_lags(csv_path: Path) -> set[int]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "lags" not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not contain a 'lags' column")
        return {
            int(float(row["lags"]))
            for row in reader
            if row.get("lags", "").strip()
        }


def expected_lags(min_lag: int, max_lag: int, step: int) -> tuple[int, ...]:
    if step <= 0:
        raise ValueError("--step must be positive")
    return tuple(range(min_lag, max_lag + 1, step))


def csvs_for_scope(run_dir: Path, scope: str) -> list[tuple[str, Path]]:
    if scope == "super_subject":
        return [("super_subject", run_dir / "lag_performance.csv")]

    csvs = []
    for csv_path in sorted(run_dir.glob("*/lag_performance.csv")):
        csvs.append((csv_path.parent.name, csv_path))
    return csvs


def find_gaps(
    config: Mapping,
    model: str,
    root: Path,
    expected: Sequence[int],
    ignored: set[int],
) -> tuple[list[Gap], list[str]]:
    result_config = config.get("results", {}).get(model)
    if result_config is None:
        raise KeyError(f"No results.{model!r} section found in config")

    expected_set = set(expected) - ignored
    gaps: list[Gap] = []
    issues: list[str] = []

    for task, scopes in result_config.items():
        for scope, paths in scopes.items():
            entity_lags: dict[str, set[int]] = defaultdict(set)
            for raw_path in as_list(paths):
                run_dir = resolve_path(raw_path, root)
                if not run_dir.exists():
                    issues.append(f"{task} | {scope} | missing directory: {raw_path}")
                    continue

                csvs = csvs_for_scope(run_dir, scope)
                existing_csvs = [(entity, csv_path) for entity, csv_path in csvs if csv_path.exists()]
                if not existing_csvs:
                    issues.append(f"{task} | {scope} | no lag_performance.csv files under {raw_path}")
                    continue

                for entity, csv_path in existing_csvs:
                    entity_lags[entity].update(read_lags(csv_path))

            for entity, lags in sorted(entity_lags.items()):
                present = lags & expected_set
                missing = tuple(sorted(expected_set - lags))
                if missing:
                    gaps.append(
                        Gap(
                            task=task,
                            scope=scope,
                            entity=entity,
                            missing=missing,
                            present_count=len(present),
                        )
                    )

    return gaps, issues


def lag_ranges(lags: Sequence[int], step: int = DEFAULT_STEP) -> tuple[tuple[int, int], ...]:
    if not lags:
        return ()

    ranges = []
    start = prev = sorted(lags)[0]
    for lag in sorted(lags)[1:]:
        if lag == prev + step:
            prev = lag
            continue
        ranges.append((start, prev))
        start = prev = lag
    ranges.append((start, prev))
    return tuple(ranges)


def format_ranges(lags: Sequence[int], step: int = DEFAULT_STEP) -> str:
    ranges = lag_ranges(lags, step)
    if not ranges:
        return "none"
    return ", ".join(str(start) if start == end else f"{start}..{end}" for start, end in ranges)


def entity_sort_key(entity: str) -> tuple[str, int | str]:
    prefix, _, suffix = entity.rpartition("_")
    if suffix.isdigit():
        return (prefix, int(suffix))
    return (entity, entity)


def print_gap_report(gaps: Sequence[Gap], total_expected: int, step: int) -> None:
    print("Gaps")
    print(f"Expected lag count: {total_expected}")
    if not gaps:
        print("No missing lags found.")
        return

    by_task_scope: dict[tuple[str, str], list[Gap]] = defaultdict(list)
    for gap in gaps:
        by_task_scope[(gap.task, gap.scope)].append(gap)

    for task, scope in sorted(by_task_scope):
        print(f"\n[{task} | {scope}]")
        grouped: dict[tuple[int, ...], list[Gap]] = defaultdict(list)
        for gap in by_task_scope[(task, scope)]:
            grouped[gap.missing].append(gap)
        for missing, members in sorted(grouped.items(), key=lambda item: (len(item[0]), item[0])):
            entities = ", ".join(sorted((gap.entity for gap in members), key=entity_sort_key))
            print(f"  {entities}: missing {len(missing)} [{format_ranges(missing, step)}]")


def safe_load_config(path: Path) -> Mapping | None:
    try:
        return load_yaml(path)
    except Exception:
        return None


def get_nested(config: Mapping, keys: Sequence[str]):
    value = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def discover_candidate_configs(root: Path) -> list[CandidateConfig]:
    candidates = []
    for path in sorted(root.rglob("*.yml")):
        config = safe_load_config(path)
        if not isinstance(config, Mapping) or "tasks" in config:
            continue

        task_name = get_nested(config, ("task_config", "task_name"))
        run_mode = str(config.get("run_mode", ""))
        trial_name = str(config.get("trial_name", ""))
        path_text = str(path)
        if not run_mode:
            if "supersubject" in path_text or "supersubject" in trial_name:
                run_mode = "supersubject"
            elif "per_subject" in path_text or "per_subject" in trial_name:
                run_mode = "per_subject"
            elif "per_region" in path_text or "per_region" in trial_name:
                run_mode = "per_region"
        if not task_name or not run_mode or not trial_name:
            continue

        candidates.append(
            CandidateConfig(
                path=path,
                task_name=str(task_name),
                run_mode=run_mode,
                trial_name=trial_name,
            )
        )
    return candidates


def config_score(candidate: CandidateConfig, task: str, scope: str) -> tuple[int, str]:
    score = 0
    name = candidate.path.name
    trial = candidate.trial_name

    if scope == "super_subject" and ("supersubject" in name or "supersubject" in trial):
        score += 3
    if scope != "super_subject" and scope in name:
        score += 3
    if task in trial:
        score += 3
    if task in str(candidate.path):
        score += 1
    if task == "word_embedding" and "gpt2" in name:
        score += 10
    if task == "volume_level" and "simple" in name:
        score += 10
    if task == "llm_decoding" and "two_stage" not in name:
        score += 10
    return (-score, str(candidate.path))


def choose_config(
    candidates: Sequence[CandidateConfig],
    task: str,
    scope: str,
) -> Path | None:
    task_name = TASK_NAMES.get(task)
    run_mode = SCOPE_TO_RUN_MODE.get(scope)
    if not task_name or not run_mode:
        return None

    matches = [
        candidate
        for candidate in candidates
        if candidate.task_name == task_name and candidate.run_mode == run_mode
    ]
    if not matches:
        return None

    return sorted(matches, key=lambda candidate: config_score(candidate, task, scope))[0].path


def subject_id(entity: str) -> str:
    return entity.removeprefix("subject_")


def region_id(entity: str) -> str:
    return entity.removeprefix("region_").upper()


def shell_join_values(values: Iterable[str]) -> str:
    return "[" + ",".join(values) + "]"


def make_commands(
    gaps: Sequence[Gap],
    candidates: Sequence[CandidateConfig],
    step: int,
) -> tuple[list[str], list[str]]:
    command_groups: dict[tuple[str, str, tuple[int, int]], list[str]] = defaultdict(list)
    config_issues: list[str] = []

    for gap in gaps:
        config = choose_config(candidates, gap.task, gap.scope)
        if config is None:
            config_issues.append(f"{gap.task} | {gap.scope} | no matching baseline config")
            continue

        for start, end in lag_ranges(gap.missing, step):
            command_groups[(str(config), gap.scope, (start, end))].append(gap.entity)

    commands = []
    for (config, scope, (start, end)), entities in sorted(command_groups.items()):
        max_lag = end + step
        overrides = [
            f"--training_params.min_lag={start}",
            f"--training_params.max_lag={max_lag}",
            f"--training_params.lag_step_size={step}",
        ]
        if scope == "per_subject":
            subjects = sorted((subject_id(entity) for entity in entities), key=int)
            overrides.append(
                f"--task_config.data_params.subject_ids={shell_join_values(subjects)}"
            )
        elif scope == "per_region":
            regions = sorted((region_id(entity) for entity in entities))
            overrides.append(f"--regions={shell_join_values(regions)}")

        commands.append(
            f"make train-config CONFIG={config} OVERRIDES=\"{' '.join(overrides)}\""
        )

    return commands, sorted(set(config_issues))


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    config = load_yaml(args.config)
    expected = expected_lags(args.min_lag, args.max_lag, args.step)
    ignored = set(args.ignore_lag)

    gaps, issues = find_gaps(config, args.model, root, expected, ignored)
    print_gap_report(gaps, len(set(expected) - ignored), args.step)

    if issues:
        print("\nData Issues")
        for issue in issues:
            print(f"  {issue}")

    if not args.no_commands:
        candidates = discover_candidate_configs(args.baseline_config_root)
        commands, config_issues = make_commands(gaps, candidates, args.step)

        if config_issues:
            print("\nCommand Issues")
            for issue in config_issues:
                print(f"  {issue}")

        print("\nSuggested Make Commands")
        if commands:
            for command in commands:
                print(command)
        else:
            print("No commands needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
