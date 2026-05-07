#!/usr/bin/env python

"""Format Makefile group specs as override values and job-name tags."""

import argparse
import ast
import re
import sys
from typing import Iterable, List, Optional


def _split_top_level(value: str, delimiter: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None

    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1

    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _split_items(value: str) -> List[str]:
    return [
        item.strip().strip("'\"")
        for item in re.split(r"[,\s]+", value.strip())
        if item.strip()
    ]


def _parse_loose_group(value: str) -> List[List[str]]:
    value = value.strip()
    if not value:
        return []

    if ";" in value:
        return [_split_items(group) for group in value.split(";") if group.strip()]

    if value.startswith("[[") and value.endswith("]]"):
        inner = value[1:-1].strip()
        groups = []
        for group in _split_top_level(inner, ","):
            group = group.strip()
            if not group.startswith("[") or not group.endswith("]"):
                raise ValueError(f"Expected bracketed group, got: {group}")
            groups.append(_split_items(group[1:-1]))
        return groups

    return [_split_items(value)]


def _parse_groups(value: str) -> List[List[object]]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return _parse_loose_group(value)

    if not isinstance(parsed, list):
        raise ValueError("Group spec must be a list or semicolon-separated groups.")

    if not parsed:
        return []

    if all(not isinstance(item, list) for item in parsed):
        return [parsed]

    if not all(isinstance(item, list) for item in parsed):
        raise ValueError("Group spec must be a list of lists.")

    return parsed


def _parse_items(value: str) -> List[str]:
    return _split_items(value)


def _chunk(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _normalize_item(item: object, kind: str) -> object:
    if kind == "int":
        try:
            return int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected integer item, got: {item!r}") from exc
    return str(item)


def _format_override(items: Iterable[object], kind: str) -> str:
    if kind == "int":
        return "[" + ",".join(str(item) for item in items) + "]"
    return "[" + ",".join(repr(str(item)) for item in items) + "]"


def _format_tag(items: Iterable[object], prefix: str) -> str:
    cleaned = []
    for item in items:
        text = re.sub(r"[^A-Za-z0-9]+", "-", str(item)).strip("-")
        cleaned.append(text)
    return prefix + "-".join(cleaned)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit Makefile-safe GROUP_VALUE|GROUP_TAG entries from explicit groups "
            "or a batch size."
        )
    )
    parser.add_argument("--groups", default="", help="Nested list or ';' separated groups.")
    parser.add_argument("--items", default="", help="Fallback whitespace/comma item list.")
    parser.add_argument("--batch-size", default="", help="Chunk fallback items by this size.")
    parser.add_argument("--kind", choices=("int", "str"), required=True)
    parser.add_argument("--tag-prefix", default="")
    args = parser.parse_args()

    try:
        if args.groups.strip():
            raw_groups = _parse_groups(args.groups)
        elif args.batch_size.strip():
            batch_size = int(args.batch_size)
            if batch_size < 1:
                raise ValueError("--batch-size must be >= 1")
            raw_groups = _chunk(_parse_items(args.items), batch_size)
        else:
            raw_groups = [[item] for item in _parse_items(args.items)]

        for raw_group in raw_groups:
            group = [_normalize_item(item, args.kind) for item in raw_group]
            if not group:
                continue
            print(f"{_format_override(group, args.kind)}|{_format_tag(group, args.tag_prefix)}")
    except ValueError as exc:
        print(f"format_make_groups.py: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
