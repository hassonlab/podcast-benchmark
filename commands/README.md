# Commands

Standard submit path:
- `./commands/submit-task.sh --model brainbert --task llm_decoding`
- `./commands/submit-task.sh --model popt --task content_noncontent --fold 1 --lag 0`
- `./commands/submit-task.sh --model diver --task volume_level --node node1`

Standard defaults:
- `training_params.fold_ids = [1,5]`
- `training_params.min_lag = -1000`
- `training_params.max_lag = 1500`
- `training_params.lag_step_size = 500`
- Fold ids are 1-based

Standard debug path:
- `./commands/debug/submit-task.sh --model brainbert --task content_noncontent --epochs 1`
- default debug overrides: `fold_ids = [1]`, `lag = 0`

Job utilities:
- `./commands/job-status.sh 94346 94349`
- `./commands/job-cancel.sh 94346 94349`

Experimental paths live under:
- `./commands/experimental/...`

Standard `submit-task.sh` always uses the normal benchmark path and normal configs.
Experimental submitters are isolated and do not affect the standard path.
