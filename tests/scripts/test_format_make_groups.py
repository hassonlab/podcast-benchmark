import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "format_make_groups.py"


def run_groups(*args):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip().splitlines()


def test_formats_loose_subject_groups():
    assert run_groups(
        "--groups",
        "[[1, 2, 3], [4, 5, 6]]",
        "--kind",
        "int",
        "--tag-prefix",
        "sub",
    ) == ["[1,2,3]|sub1-2-3", "[4,5,6]|sub4-5-6"]


def test_formats_loose_region_groups():
    assert run_groups(
        "--groups",
        "[[EAC, ITG, MTG], [IFG, TP, TPJ]]",
        "--kind",
        "str",
    ) == ["['EAC','ITG','MTG']|EAC-ITG-MTG", "['IFG','TP','TPJ']|IFG-TP-TPJ"]


def test_chunks_items_when_batch_size_is_set():
    assert run_groups(
        "--items",
        "1 3 4 5 6 7 8 9",
        "--batch-size",
        "3",
        "--kind",
        "int",
        "--tag-prefix",
        "sub",
    ) == ["[1,3,4]|sub1-3-4", "[5,6,7]|sub5-6-7", "[8,9]|sub8-9"]
