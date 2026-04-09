# Experimental Lazy Volume Level

This subtree isolates a lazy-STFT experiment for large `volume_level` runs.

Design:
- base files are copied from the standard path and modified only here
- only `volume_level + {brainbert,popt}` uses the lazy runner
- all other tasks go through the standard pipeline, even when launched from this subtree
- CLI overrides must use `--key=value` form, e.g. `--training_params.lag=0`

Rollback:
```bash
rm -rf /storage/connectome/jmhan/podcast-benchmark/experimental/lazy_volume_level
rm -rf /storage/connectome/jmhan/podcast-benchmark/commands/experimental/lazy_volume_level
rm -rf /storage/connectome/jmhan/podcast-benchmark/configs/experimental/lazy_volume_level
```
