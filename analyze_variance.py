import os
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from main import load_config
from data_utils import load_raws, load_word_data, get_data
import registry


def analyze_embedding_variance(trial_name: str, batch_size: int, output_dir: str):
    # Load config from standard location
    config_path = os.path.join("results", trial_name, "config.yml")
    experiment_config = load_config(config_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    raws = load_raws(experiment_config.data_params)
    df_word = load_word_data(experiment_config.data_params)

    # Apply optional config setter
    if experiment_config.config_setter_name:
        setter = registry.config_setter_registry[experiment_config.config_setter_name]
        experiment_config = setter(experiment_config, raws, df_word)

    # Optional preprocessing
    preprocessing_fn = None
    if experiment_config.data_params.preprocessing_fn_name:
        preprocessing_fn = registry.data_preprocessor_registry[
            experiment_config.data_params.preprocessing_fn_name
        ]

    model_ctor = registry.model_constructor_registry[
        experiment_config.model_constructor_name
    ]
    trial_dir = os.path.join("models", trial_name)
    lag_folders = sorted(os.listdir(trial_dir), key=lambda fn: int(fn.split("_")[-1]))
    lags = [int(fn.split("_")[-1]) for fn in lag_folders]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for lag in lags:
        print(f"--- analyzing lag {lag} ---")
        X, Y, _ = get_data(
            lag,
            raws,
            df_word,
            window_width=experiment_config.data_params.window_width,
            preprocessing_fn=preprocessing_fn,
            preprocessor_params=experiment_config.data_params.preprocessor_params,
        )
        X = torch.FloatTensor(X)

        kf = KFold(n_splits=5, shuffle=False)
        all_embeddings = []

        for fold, (_, test_idx) in enumerate(kf.split(X), start=1):
            ckpt_path = os.path.join(
                trial_dir, f"lag_{lag}", f"best_model_fold{fold}.pt"
            )
            model = model_ctor(experiment_config.model_params)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True),
                strict=False,
            )
            model.to(device).eval()

            loader = DataLoader(
                TensorDataset(X[test_idx]), batch_size=batch_size, shuffle=False
            )
            fold_feats = []

            with torch.no_grad():
                for xb in loader:
                    xb = xb[0].to(device)
                    feats = model(xb)
                    fold_feats.append(feats.cpu().numpy())

            all_embeddings.append(np.concatenate(fold_feats, axis=0))

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
        var_by_dim = np.var(all_embeddings, axis=0)
        mean_var = var_by_dim.mean()

        print(
            f"lag {lag:>4} → embeddings.shape = {all_embeddings.shape}, mean variance = {mean_var:.4e}"
        )

        plt.figure(figsize=(6, 3))
        plt.hist(var_by_dim, bins=30, edgecolor="k")
        plt.title(f"Embedding‐dim variances (lag={lag})\nmean var = {mean_var:.4e}")
        plt.xlabel("Variance of dimension")
        plt.ylabel("Count")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"lag_{lag}_variance.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved variance plot to {plot_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze embedding variance across lags and folds"
    )
    parser.add_argument(
        "--trial", type=str, required=True, help="Name of trial folder under models/"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Directory to save plots"
    )
    args = parser.parse_args()

    analyze_embedding_variance(
        trial_name=args.trial, batch_size=args.batch_size, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
