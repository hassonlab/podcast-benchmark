import itertools
import argparse
import subprocess
import shlex
import yaml
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Launch Slurm jobs for hyperparameter search."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="YAML file specifying parameter grid.",
    )
    parser.add_argument(
        "--param",
        action="append",
        nargs="+",
        metavar=("key", "val1", "val2", "..."),
        help="Additional parameter and its values to override or add, e.g. --param lr 0.001 0.01",
    )
    parser.add_argument(
        "--submit_script",
        type=str,
        default="submit.sh",
        help="Path to the Slurm submission script.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print sbatch commands without executing.",
    )
    return parser.parse_args()


def load_param_grid(config_path: str) -> dict:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return raw.get("param_grid", raw)


def main():
    args = parse_arguments()

    # Load from YAML config
    param_dict = load_param_grid(args.config) if args.config else {}

    # Apply CLI overrides
    if args.param:
        for entry in args.param:
            key = entry[0]
            values = entry[1:]
            param_dict[key] = values

    if not param_dict:
        raise ValueError(
            "No parameters provided. Use --config or --param to specify search space."
        )

    # Generate combinations
    keys = list(param_dict.keys())
    value_combinations = list(itertools.product(*[param_dict[k] for k in keys]))

    print(f"Launching {len(value_combinations)} jobs...")

    for combo in value_combinations:
        combo_dict = dict(zip(keys, combo))

        # Construct job name and export variables
        job_name = "-".join(f"{k}={str(v)}" for k, v in combo_dict.items())
        env_vars = ",".join(
            f"{k.upper()}={shlex.quote(str(v))}" for k, v in combo_dict.items()
        )

        cmd = (
            f"sbatch --job-name={job_name} --export=ALL,{env_vars} {args.submit_script}"
        )

        if args.dry_run:
            print(cmd)
        else:
            subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
