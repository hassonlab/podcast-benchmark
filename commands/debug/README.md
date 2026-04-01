# Debug Commands

This subtree is for debugging-only runs on the standard benchmark path.

Default debug policy:
- `fold_ids = [1]`
- `lag = 0`

Use:
- `./commands/debug/submit-task.sh --model brainbert --task content_noncontent --epochs 1`

This path still uses:
- `configs/foundation_models/...`
- root `main.py`
