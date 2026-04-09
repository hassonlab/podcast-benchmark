"""Lightweight terminal formatting (no extra dependencies)."""

from __future__ import annotations

import shutil
import sys
from typing import Iterable, Optional, Sequence


def _width() -> int:
    try:
        return min(max(shutil.get_terminal_size(fallback=(72, 20)).columns, 52), 96)
    except Exception:
        return 72


def dim(s: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[2m{s}\033[0m"


def bold(s: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[1m{s}\033[0m"


def green(s: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[32m{s}\033[0m"


def cyan(s: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[36m{s}\033[0m"


def rule(char: str = "─", title: str | None = None) -> None:
    w = _width()
    if title:
        pad = max(2, w - len(title) - 4)
        left = pad // 2
        right = pad - left
        print(f"{char * left} {bold(title)} {char * right}")
    else:
        print(char * w)


def banner(subtitle: str | None = None) -> None:
    w = _width()
    title = "Raider temporal VAE"
    if subtitle:
        title = f"{title} · {subtitle}"
    line = f" {title} "
    print()
    print(cyan("╭" + "─" * (w - 2) + "╮"))
    pad = w - 4 - len(line)
    if pad < 0:
        line = line[: w - 6] + "… "
        pad = w - 4 - len(line)
    print(cyan("│") + bold(line) + " " * pad + cyan("│"))
    print(cyan("╰" + "─" * (w - 2) + "╯"))


def key_value_block(rows: Sequence[tuple[str, str]], indent: str = "  ") -> None:
    if not rows:
        return
    klen = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{indent}{k.ljust(klen)}  {dim('·')}  {v}")


def metrics_table(headers: tuple[str, ...], rows: Iterable[tuple]) -> None:
    rows = list(rows)
    if not rows:
        return
    cols = list(zip(*rows))
    widths = [max(len(str(h)), max(len(str(c[i])) for c in rows)) for i, h in enumerate(headers)]
    sep = " │ "
    head = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print()
    print(dim("  " + "─" * (len(head) + 4)))
    print("  " + bold(sep.join(headers[i].ljust(widths[i]) for i in range(len(headers)))))
    print(dim("  " + "─" * (len(head) + 4)))
    for row in rows:
        print("  " + sep.join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    print(dim("  " + "─" * (len(head) + 4)))


def epoch_line(
    epoch: int,
    n_epochs: int,
    loss: float,
    l_recon: float,
    l_cross: float,
    l_kl: float,
    l_other: Optional[float] = None,
    val_loss: Optional[float] = None,
    l_smooth: Optional[float] = None,
) -> None:
    bar_w = 18
    filled = int(bar_w * epoch / max(n_epochs, 1))
    bar = "█" * filled + "░" * (bar_w - filled)
    pct = 100.0 * epoch / max(n_epochs, 1)
    o = f"  {dim('O')}{l_other:.4f}" if l_other is not None else ""
    v = f"  {dim('V')}{val_loss:.4f}" if val_loss is not None else ""
    s = f"  {dim('S')}{l_smooth:.4f}" if l_smooth is not None else ""
    print(
        f"  {bar} {cyan(f'{pct:5.1f}%')}  "
        f"{bold(f'{epoch:>3}/{n_epochs}')}  "
        f"{dim('L')}{loss:.4f}  "
        f"{dim('R')}{l_recon:.4f}  "
        f"{dim('X')}{l_cross:.4f}  "
        f"{dim('K')}{l_kl:.4f}"
        f"{o}"
        f"{s}"
        f"{v}"
    )


def done_line(msg: str) -> None:
    print()
    print(f"  {green('✓')} {msg}")
