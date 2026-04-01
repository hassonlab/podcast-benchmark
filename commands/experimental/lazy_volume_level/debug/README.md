# Experimental Lazy Volume Level Debug Commands

This subtree is for debugging-only runs on the experimental lazy-volume path.

Default debug policy:
- `fold_ids = [1]`
- `lag = 0`

Use:
- `./commands/experimental/lazy_volume_level/debug/submit-task.sh --model popt --task volume_level --epochs 1 --node node1`

This path uses:
- `configs/experimental/lazy_volume_level/...`
- `experimental/lazy_volume_level/main.py`
