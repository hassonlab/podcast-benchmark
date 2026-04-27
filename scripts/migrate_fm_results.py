#!/usr/bin/env python3
"""Normalize old foundation-model result runs into the current results layout."""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$")
SUBJECT_RE = re.compile(r"subject(\d+)_full$")


@dataclass(frozen=True)
class Candidate:
    model: str
    task: str
    condition: str
    subject: Optional[int]
    timestamp: str
    source: Path
    dest: Path
    selected: bool
    action: str
    reason: str


def parse_timestamp(run_dir: Path) -> Optional[str]:
    match = TIMESTAMP_RE.search(run_dir.name)
    return match.group(1) if match else None


def is_valid_run(run_dir: Path) -> bool:
    return (run_dir / "lag_performance.csv").exists()


def copy_or_move_run(source: Path, dest: Path, move: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(source), str(dest))
    else:
        shutil.copytree(source, dest, dirs_exist_ok=True)


def discover_mappings(source: Path, dest_root: Path) -> List[Candidate]:
    mappings: List[Candidate] = []
    for model_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        model = model_dir.name
        for task_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            task = task_dir.name
            mappings.extend(discover_super_subject(model, task, task_dir, dest_root))
            mappings.extend(discover_per_subject(model, task, task_dir, dest_root))
    return mappings


def discover_super_subject(model: str, task: str, task_dir: Path, dest_root: Path) -> List[Candidate]:
    mode_dir = task_dir / "persubject_concat"
    if not mode_dir.exists():
        return []

    runs = []
    for run_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
        timestamp = parse_timestamp(run_dir)
        if timestamp and is_valid_run(run_dir):
            runs.append((timestamp, run_dir))

    selected = max(runs, default=None, key=lambda item: item[0])
    mappings = []
    for timestamp, run_dir in runs:
        is_selected = selected is not None and run_dir == selected[1]
        dest = dest_root / f"{model}_{task}_super_subject_{timestamp}"
        mappings.append(
            Candidate(
                model=model,
                task=task,
                condition="super_subject",
                subject=None,
                timestamp=timestamp,
                source=run_dir,
                dest=dest,
                selected=is_selected,
                action="copy" if is_selected else "skip",
                reason="newest valid run" if is_selected else "older run skipped",
            )
        )
    return mappings


def discover_per_subject(model: str, task: str, task_dir: Path, dest_root: Path) -> List[Candidate]:
    mode_dir = task_dir / "subject_full"
    if not mode_dir.exists():
        return []

    selected_by_subject = {}
    runs_by_subject = {}
    for subject_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
        subject_match = SUBJECT_RE.match(subject_dir.name)
        if not subject_match:
            continue
        subject = int(subject_match.group(1))
        runs = []
        for run_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            timestamp = parse_timestamp(run_dir)
            if timestamp and is_valid_run(run_dir):
                runs.append((timestamp, run_dir))
        if runs:
            runs_by_subject[subject] = runs
            selected_by_subject[subject] = max(runs, key=lambda item: item[0])

    if not selected_by_subject:
        return []

    batch_timestamp = min(timestamp for timestamp, _run_dir in selected_by_subject.values())
    dest_run = dest_root / f"{model}_{task}_per_subject_{batch_timestamp}"

    mappings = []
    for subject, runs in sorted(runs_by_subject.items()):
        selected = selected_by_subject[subject][1]
        for timestamp, run_dir in runs:
            is_selected = run_dir == selected
            mappings.append(
                Candidate(
                    model=model,
                    task=task,
                    condition="per_subject",
                    subject=subject,
                    timestamp=timestamp,
                    source=run_dir,
                    dest=dest_run / f"subject_{subject}",
                    selected=is_selected,
                    action="copy" if is_selected else "skip",
                    reason="newest valid subject run" if is_selected else "older subject run skipped",
                )
            )
    return mappings


def write_report(mappings: Iterable[Candidate], report_path: Path) -> pd.DataFrame:
    rows = []
    for item in mappings:
        rows.append(
            {
                "model": item.model,
                "task": item.task,
                "condition": item.condition,
                "subject": item.subject if item.subject is not None else "",
                "timestamp": item.timestamp,
                "source": str(item.source),
                "dest": str(item.dest),
                "selected": item.selected,
                "action": item.action,
                "reason": item.reason,
            }
        )
    df = pd.DataFrame(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_path, index=False)
    return df


def migrate(source: Path, dest: Path, dry_run: bool = False, move: bool = False) -> pd.DataFrame:
    mappings = discover_mappings(source, dest)
    action = "move" if move else "copy"
    mappings = [
        Candidate(
            **{
                **item.__dict__,
                "action": action if item.selected else "skip",
            }
        )
        for item in mappings
    ]

    report = write_report(mappings, dest / "migration_report.csv")
    selected = [item for item in mappings if item.selected]

    for item in selected:
        print(f"{item.action}: {item.source} -> {item.dest}")
        if not dry_run:
            copy_or_move_run(item.source, item.dest, move=move)

    skipped = len(mappings) - len(selected)
    print(f"planned {len(selected)} selected mappings; skipped {skipped} older runs")
    if dry_run:
        print(f"dry run only; report written to {dest / 'migration_report.csv'}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results-fm/foundation_models", type=Path)
    parser.add_argument("--dest", default="results-fm-normalized", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--move", action="store_true", help="Move runs instead of copying them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate(args.source, args.dest, dry_run=args.dry_run, move=args.move)


if __name__ == "__main__":
    main()
