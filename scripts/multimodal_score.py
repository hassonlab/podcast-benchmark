#!/usr/bin/env python3
"""
Compute a multimodal decoding score from word embedding and whisper embedding results.

Combines word pairwise accuracy and whisper pairwise accuracy using the harmonic mean.
Both metrics have chance level 0.5 and range [0, 1], so the harmonic mean
is well-calibrated (chance-level inputs yield a chance-level score).

Usage:
    python scripts/multimodal_score.py results/word_run_2025-... results/whisper_run_2025-...
    python scripts/multimodal_score.py results/word_run results/whisper_run --plot
    python scripts/multimodal_score.py results/word_run results/whisper_run -o combined.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def harmonic_mean(a, b):
    """Compute element-wise harmonic mean of two Series."""
    return 2 * (a * b) / (a + b)


def main():
    parser = argparse.ArgumentParser(
        description="Compute multimodal decoding score (harmonic mean of word and whisper pairwise accuracy)."
    )
    parser.add_argument(
        "word_dir",
        type=Path,
        help="Path to word embedding result directory (contains lag_performance.csv)",
    )
    parser.add_argument(
        "whisper_dir",
        type=Path,
        help="Path to whisper embedding result directory (contains lag_performance.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to save combined CSV (default: <word_dir>/multimodal_score.csv)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot multimodal score vs lag",
    )
    args = parser.parse_args()

    # Read lag_performance.csv from each directory
    word_csv = args.word_dir / "lag_performance.csv"
    whisper_csv = args.whisper_dir / "lag_performance.csv"

    if not word_csv.exists():
        print(f"Error: {word_csv} not found", file=sys.stderr)
        sys.exit(1)
    if not whisper_csv.exists():
        print(f"Error: {whisper_csv} not found", file=sys.stderr)
        sys.exit(1)

    word_df = pd.read_csv(word_csv)
    whisper_df = pd.read_csv(whisper_csv)

    # Validate required columns
    word_metric = "test_pairwise_accuracy_mean"
    whisper_metric = "test_pairwise_accuracy_mean"

    if word_metric not in word_df.columns:
        print(f"Error: column '{word_metric}' not found in {word_csv}", file=sys.stderr)
        print(f"  Available columns: {list(word_df.columns)}", file=sys.stderr)
        sys.exit(1)
    if whisper_metric not in whisper_df.columns:
        print(f"Error: column '{whisper_metric}' not found in {whisper_csv}", file=sys.stderr)
        print(f"  Available columns: {list(whisper_df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Join on lags — rename to distinguish since both use the same metric name
    word_col = "word_pairwise_accuracy"
    whisper_col = "whisper_pairwise_accuracy"
    word_subset = word_df[["lags", word_metric]].rename(columns={word_metric: word_col})
    whisper_subset = whisper_df[["lags", whisper_metric]].rename(columns={whisper_metric: whisper_col})
    merged = pd.merge(word_subset, whisper_subset, on="lags", how="inner")

    if merged.empty:
        print("Error: no matching lags between the two result sets", file=sys.stderr)
        sys.exit(1)

    # Compute harmonic mean
    merged["multimodal_score"] = harmonic_mean(
        merged[word_col], merged[whisper_col]
    )

    # Print summary table
    print()
    print("Multimodal Decoding Score")
    print("=" * 65)
    print(f"  Word results:    {args.word_dir}")
    print(f"  Whisper results: {args.whisper_dir}")
    print()
    print(f"{'Lag':>6}  {'Word Pairwise':>14}  {'Whisper Pairwise':>17}  {'Multimodal':>11}")
    print("-" * 65)
    for _, row in merged.iterrows():
        print(
            f"{int(row['lags']):>6}  "
            f"{row[word_col]:>14.4f}  "
            f"{row[whisper_col]:>17.4f}  "
            f"{row['multimodal_score']:>11.4f}"
        )
    print("-" * 65)

    # Best lag
    best_idx = merged["multimodal_score"].idxmax()
    best_row = merged.loc[best_idx]
    print(
        f"\nBest multimodal score: {best_row['multimodal_score']:.4f} "
        f"at lag {int(best_row['lags'])}ms"
    )

    # Save CSV
    output_path = args.output or (args.word_dir / "multimodal_score.csv")
    merged.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Optional plot
    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            merged["lags"], merged[word_col],
            marker="o", linewidth=2, markersize=6, label="Word Pairwise Acc",
        )
        ax.plot(
            merged["lags"], merged[whisper_col],
            marker="s", linewidth=2, markersize=6, label="Whisper Pairwise Acc",
        )
        ax.plot(
            merged["lags"], merged["multimodal_score"],
            marker="D", linewidth=2, markersize=6, label="Multimodal Score",
        )
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
        ax.axvline(
            best_row["lags"], color="red", linestyle="--", alpha=0.5,
            label=f"Best lag: {int(best_row['lags'])}ms",
        )

        ax.set_xlabel("Lag (ms)", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            f"Multimodal Decoding Score\n"
            f"Best: {best_row['multimodal_score']:.4f} at {int(best_row['lags'])}ms",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()

        plot_path = output_path.with_suffix(".png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
