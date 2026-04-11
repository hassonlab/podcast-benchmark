#!/usr/bin/env python3
"""
Run **tutorial 11 SRM** on Pieman ``movie.npy``, then **train both temporal VAE checkpoints** (TSM +
full-movie) and **eval** — one command for the full Pieman numpy pipeline (parallel to Raider).

Steps (in order):

1. ``run_srm_tutorial.py`` — BrainIAK SRM, ISC, time-segment matching (figures under ``<out>/srm``).
2. ``train_temporal_vae.py`` — first-half checkpoint (TSM eval).
3. ``train_temporal_vae.py`` — full-movie checkpoint (image LOO eval, if you add image/label later).
4. ``eval_temporal_vae.py`` — TSM + image sections with ``--checkpoint-tsm`` and ``--checkpoint-image``.

Requires ``data/pieman/movie.npy``. Pieman VAE configs use a **small** ``enc_ch``/``dec_ch`` and
``batch_size: 1`` because ~98k voxels × 18 subjects is huge; if training still OOMs, rerun with
``--device cpu``.

Usage::

  python fmri_pieman/run_srm_and_vae.py

  python fmri_pieman/run_srm_and_vae.py --data-dir data/pieman --out-dir results/pieman_full --device cuda

  python fmri_pieman/run_srm_and_vae.py --skip-srm --skip-vae-train   # only eval (needs checkpoints)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(step: str, cmd: list[str]) -> None:
    tqdm.write("")
    tqdm.write("=" * 72)
    tqdm.write(f"  {step}")
    tqdm.write("=" * 72)
    tqdm.write("  " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pieman: SRM (tutorial 11) + temporal VAE train + eval in one command"
    )
    p.add_argument(
        "--data-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "pieman"),
        help="Directory with movie.npy",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(_PROJECT_ROOT, "results", "pieman_srm_vae"),
        help="SRM figures go to <out-dir>/srm; checkpoints use config paths",
    )
    p.add_argument(
        "--config-tsm",
        default="fmri_pieman/configs/pieman_temporal_vae.yml",
        help="YAML for first-half (TSM) VAE",
    )
    p.add_argument(
        "--config-image",
        default="fmri_pieman/configs/pieman_temporal_vae_full_movie.yml",
        help="YAML for full-movie VAE (image checkpoint)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu", "auto"),
        help="Device for VAE train/eval (default: cuda)",
    )
    p.add_argument("--skip-srm", action="store_true", help="Skip BrainIAK SRM tutorial step")
    p.add_argument(
        "--skip-vae-train",
        action="store_true",
        help="Skip both VAE training steps (still run eval if checkpoints exist)",
    )
    p.add_argument("--skip-eval", action="store_true", help="Skip eval_temporal_vae.py")
    p.add_argument(
        "--run-image-srm",
        action="store_true",
        help="Pass --run-image to run_srm_tutorial if image.npy/label.npy exist",
    )
    args = p.parse_args()

    data_dir = os.path.abspath(
        args.data_dir if os.path.isabs(args.data_dir) else os.path.join(_PROJECT_ROOT, args.data_dir)
    )
    out_dir = os.path.abspath(
        args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_PROJECT_ROOT, args.out_dir)
    )
    srm_out = os.path.join(out_dir, "srm")

    py = sys.executable
    cfg_tsm = args.config_tsm
    cfg_img = args.config_image
    if not os.path.isabs(cfg_tsm):
        cfg_tsm = os.path.join(_PROJECT_ROOT, cfg_tsm)
    if not os.path.isabs(cfg_img):
        cfg_img = os.path.join(_PROJECT_ROOT, cfg_img)

    ckpt_tsm = os.path.join(_PROJECT_ROOT, "fmri_pieman", "checkpoints", "pieman_temporal_vae.pt")
    ckpt_img = os.path.join(
        _PROJECT_ROOT, "fmri_pieman", "checkpoints", "pieman_temporal_vae_full_movie.pt"
    )

    steps: list[tuple[str, list[str]]] = []

    if not args.skip_srm:
        cmd_srm = [
            py,
            os.path.join(_PROJECT_ROOT, "fmri_pieman", "run_srm_tutorial.py"),
            "--data-dir",
            data_dir,
            "--out-dir",
            srm_out,
        ]
        if args.run_image_srm:
            cmd_srm.append("--run-image")
        steps.append(("SRM (tutorial 11)", cmd_srm))

    if not args.skip_vae_train:
        steps.append(
            (
                "VAE train · first half (TSM ckpt)",
                [
                    py,
                    os.path.join(_PROJECT_ROOT, "fmri_pieman", "train_temporal_vae.py"),
                    "--config",
                    cfg_tsm,
                    "--device",
                    args.device,
                ],
            )
        )
        steps.append(
            (
                "VAE train · full movie (image ckpt)",
                [
                    py,
                    os.path.join(_PROJECT_ROOT, "fmri_pieman", "train_temporal_vae.py"),
                    "--config",
                    cfg_img,
                    "--device",
                    args.device,
                ],
            )
        )

    if not args.skip_eval:
        steps.append(
            (
                "VAE eval (TSM + image)",
                [
                    py,
                    os.path.join(_PROJECT_ROOT, "fmri_pieman", "eval_temporal_vae.py"),
                    "--checkpoint-tsm",
                    ckpt_tsm,
                    "--checkpoint-image",
                    ckpt_img,
                    "--data-dir",
                    data_dir,
                    "--device",
                    args.device,
                ],
            )
        )

    if not steps:
        print("Nothing to run (all steps skipped).", file=sys.stderr)
        sys.exit(0)

    with tqdm(
        total=len(steps),
        desc="Pieman SRM+VAE",
        unit="step",
        file=sys.stdout,
        smoothing=0.0,
    ) as pbar:
        for desc, cmd in steps:
            pbar.set_postfix_str(desc[:56] + ("…" if len(desc) > 56 else ""), refresh=True)
            _run(desc, cmd)
            pbar.update(1)

    tqdm.write("")
    tqdm.write("All steps finished.")
    tqdm.write(f"  SRM figures: {srm_out}")
    tqdm.write(f"  Checkpoints: {os.path.dirname(ckpt_tsm)}")


if __name__ == "__main__":
    main()
