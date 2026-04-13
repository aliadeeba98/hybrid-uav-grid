# Run logs

Captured **stdout** from local validation runs. Paths are referenced from [`TESTING_README.md`](../TESTING_README.md).

| Directory | Purpose |
|-----------|---------|
| [`smoke/`](smoke/) | `py_compile` + [`scripts/smoke_train.py`](../scripts/smoke_train.py) + short CLI smoke (e.g. 20/10 episodes) |
| [`benchmarks/`](benchmarks/) | Longer train/test runs (e.g. 400/80, 800/100) with fixed seeds |
| [`archive/`](archive/) | Older benchmark logs kept for comparison |

**Regenerate** (from repo root `hybrid-uav-grid/`):

```bash
python3 -m multi_uav_grid --train-episodes 400 --test-episodes 80 --log-every 20 --log-level INFO --seed 42 \
  | tee logs/benchmarks/test_run_seed42_rerun.log
```

Use a new filename (include date or git SHA) when you do not want to overwrite a committed baseline.
