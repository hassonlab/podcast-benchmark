# SLURM GPU Usage Policy

## Scope

This document records the current GPU occupation and SLURM submission policy for the lab server, based on the user-provided policy update.

The purpose of this file is operational:

- to choose the correct partition when submitting jobs
- to avoid violating user-level QOS limits
- to distinguish safe jobs from preemptable jobs
- to guide future `sbatch` and `srun` commands issued for this repository

As recorded on: 2026-03-19

## Cluster Policy Summary

### 1. Partition overview

There are three effective queue behaviors to keep in mind.

| Partition | Intended use | Priority | Time limit | Notes |
| --- | --- | --- | --- | --- |
| `interactive` | interactive sessions via `srun`, debugging | high | max 8 hours | `srun --pty` is forced here |
| `debug` | ordinary batch jobs via `sbatch`, GPU 1 to 4 | high | unlimited | safe queue for up to 4 GPUs per user |
| `preemptable` | large batch jobs via `sbatch`, GPU 5+ | low | unlimited | may be preempted and requeued |

### 2. Automatic interactive redirection

If a user launches an interactive shell such as:

```bash
srun --pty /bin/bash
```

then the session is automatically redirected to the `interactive` partition, regardless of any manually specified partition, and is capped at 8 hours.

Practical consequence:

- use `srun` only for debugging or short interactive work
- do not expect `srun` to remain on `debug`
- do not rely on `srun` for long-running training

### 3. User-level QOS limits

The following hard limits apply per user across the server:

- maximum available GPUs: 4 guaranteed GPUs per user across all nodes
- maximum concurrent running jobs: 25
- maximum total submitted jobs: 100, including pending jobs
- maximum CPUs per GPU: 12

### 4. CPU and memory ratio rules

The cluster enforces strict resource ratios.

- CPUs per GPU: at most 12 CPU cores per requested GPU
- memory per CPU: at most 5 GB per CPU core

Examples:

- GPU 1 -> CPU must be 12 or less
- GPU 2 -> CPU must be 24 or less
- GPU 4 -> CPU must be 48 or less

Memory examples:

- CPU 4 -> memory must be 20 GB or less
- CPU 8 -> memory must be 40 GB or less
- CPU 48 -> memory must be 240 GB or less

If CPU is over-requested for the GPU count, the system may reduce it automatically to the 12x ratio.

If memory exceeds the allowed `5 GB * CPU count`, the job can be rejected.

### 5. Guaranteed versus preemptable GPU usage

#### Guaranteed region

GPU requests of 1 to 4 are treated as protected usage:

- `interactive` for `srun`
- `debug` for `sbatch`

These are the safe modes for ordinary work.

#### Preemptable region

GPU requests of 5 or more are considered opportunistic:

- the system will push them into `preemptable`
- they can be stopped and moved back to pending if higher-priority jobs need GPUs
- they should be treated as requeueable jobs

Practical consequence:

- any job using 5+ GPUs must have checkpoint save and resume logic
- do not use 5+ GPU jobs for work that cannot tolerate interruption

## Correct Submission Patterns

### A. Interactive debugging

Use for:

- quick inspection
- manual debugging
- short validation runs

Example:

```bash
srun --gres=gpu:1 --cpus-per-task=8 --pty /bin/bash
```

Notes:

- this will be forced into `interactive`
- maximum lifetime is 8 hours

### B. Ordinary training with 4 GPUs or fewer

Use for:

- normal experiments
- stable training runs
- jobs that should not be preempted

Example:

```bash
sbatch -p debug --gres=gpu:4 --cpus-per-task=48 my_script.sh
```

Notes:

- use `debug` explicitly for clarity
- stay within 1 to 4 GPUs if uninterrupted execution matters

### C. Large-scale opportunistic training with 5 GPUs or more

Use for:

- large jobs that can tolerate interruption
- throughput-oriented runs with checkpoint resume logic

Example:

```bash
sbatch --gres=gpu:6 --cpus-per-task=72 my_script.sh
```

Notes:

- the system may move this into `preemptable`
- checkpointing is mandatory in practice

### D. CPU-only jobs

Use for:

- preprocessing
- analysis
- lightweight data conversion

Example:

```bash
sbatch --cpus-per-task=32 my_cpu_script.sh
```

Notes:

- do not request GPUs unless needed
- CPU-only jobs are routed separately

## Operational Rules For This Repository

These are the repo-specific rules that should be followed when submitting jobs for `podcast-benchmark`.

### 1. Default job mode

Unless there is a strong reason otherwise:

- use `sbatch -p debug`
- use 1 to 4 GPUs only
- keep CPU requests within `12 * num_gpus`

This is the default safe mode for real training jobs in this repository.

### 2. Default debug mode

For debugging:

- use `srun --pty`
- assume an 8-hour hard limit
- do not start long training jobs interactively

### 3. Avoid accidental preemptable jobs

Do not request 5+ GPUs unless all of the following are true:

- the run can be interrupted
- checkpoint save exists
- resume logic exists
- losing temporary GPU access is acceptable

### 4. Keep user-level limits in mind before submission

Before submitting new jobs, check:

- current running GPU count
- current running job count
- current pending plus running job count

Practical implication for automation:

- do not blindly submit many experiments at once
- prefer bounded batches of jobs

### 5. CPU and memory requests must be computed together

When preparing a job:

1. choose GPU count
2. cap CPUs at `12 * GPU count`
3. cap memory at `5 GB * CPU count`

If the job is CPU-only:

- request only the CPU count actually needed
- ensure memory remains within the CPU-based cap

### 6. Recommended submission patterns for this repo

#### Small single-job experiment

```bash
sbatch -p debug --gres=gpu:1 --cpus-per-task=8 run.sh
```

#### Medium experiment

```bash
sbatch -p debug --gres=gpu:2 --cpus-per-task=24 run.sh
```

#### Large but still protected experiment

```bash
sbatch -p debug --gres=gpu:4 --cpus-per-task=48 run.sh
```

#### Interactive debug shell

```bash
srun --gres=gpu:1 --cpus-per-task=8 --pty /bin/bash
```

#### Opportunistic large run

```bash
sbatch --gres=gpu:6 --cpus-per-task=72 run.sh
```

Only use the last pattern if resume from checkpoint is already verified.

## Decision Guide

Use the following rule when choosing how to submit.

- Need a shell and manual debugging -> `srun`, expect `interactive`, max 8 hours
- Need a normal training run with 1 to 4 GPUs -> `sbatch -p debug`
- Need more than 4 GPUs -> assume preemptable behavior and require checkpoint resume
- Need preprocessing only -> CPU-only `sbatch`

## Submission Checklist

Before launching any future job from this repo, verify:

- partition choice is correct
- GPU count is intentional
- CPU request is within the 12x GPU ratio
- memory request is within the 5 GB per CPU ratio
- total user GPU usage will not exceed 4 in protected mode
- job count will not exceed the running and total submission limits
- checkpoint save and resume are ready if requesting 5+ GPUs

## Notes For Future Automation

When constructing automated submission commands for this repository:

- prefer `debug` for all non-interactive training up to 4 GPUs
- do not emit `srun` for long experiments
- avoid bulk-submitting more jobs than the user-level caps allow
- treat 5+ GPU jobs as requeueable and checkpoint-dependent by default
