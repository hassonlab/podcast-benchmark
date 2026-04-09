# Experimental Lazy Volume Level Commands

This subtree is isolated from the standard submit path.

Use:
- `./commands/experimental/lazy_volume_level/submit-task.sh --model popt --task volume_level --node node1`
- `./commands/experimental/lazy_volume_level/debug/submit-task.sh --model popt --task volume_level --epochs 1 --node node1`

Standard defaults:
- `training_params.fold_ids = [1,5]`
- `training_params.min_lag = -1000`
- `training_params.max_lag = 1500`
- `training_params.lag_step_size = 500`

Debug defaults:
- `training_params.fold_ids = [1]`
- `training_params.lag = 0`

Standard `./commands/submit-task.sh` does not use this path.
