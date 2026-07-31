# CADEngBench-A Seven-Family Submission Sample

This compact development-only sample contains one representative item for each supported joint family, the exact A1/A2 prompts, and development labels for evaluator reproduction. It contains no held-out test labels.

Run a no-cost packaging check from the repository root:

```bash
python scripts/run_cadengbench_a_benchmark.py \
  --dataset-root submission/cadengbench_a_sample7_v2 \
  --labels-root submission/cadengbench_a_sample7_v2/labels \
  --provider openai --model YOUR_MODEL --dry-run \
  --output-dir outputs/sample7_dry_run
```
