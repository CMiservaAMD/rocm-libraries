# sdpa_case_flow_poc

This package turns the SDPA forward test-plan model into a case list. The case
list is one CSV file, `levels.csv`. Each row is one test case. The row gives
the case's tier, its sizes, its estimated time and one value for every setting
of the model.

The package is a proof of concept. The design and the files are described in
[SDPA Forward Test Plan](../../docs/test-plans/sdpa-fwd.md).

## Inputs

All inputs are in [`test-plans/sdpa-fwd/`](../../test-plans/sdpa-fwd/):

| File | What it holds |
|---|---|
| `model.yaml` | The settings, their values and labels, the tier recipe and the cost model. |
| `rules.yaml` | The rules that remove combinations. |
| `injections.yaml` | Cases outside the generated sets. The generator checks that they name known values. |
| `timings-gfx942.csv`, `timings-gfx950.csv` | Measured seconds per case (`case_key,seconds,source`). A case that is not in a file gets the time from the cost model. |
| `levels.csv.sha256` | The SHA-256 of the `levels.csv` that these files give. |

The seed is 2893.

## Regenerate the CSV

Run from `dnn-providers/integration-tests/scripts`. PyYAML is required.

```bash
python3 -m sdpa_case_flow_poc.levels \
    --model-dir ../test-plans/sdpa-fwd \
    --out-dir <output folder> \
    --seed 2893 \
    --timings gfx942=../test-plans/sdpa-fwd/timings-gfx942.csv \
    --timings gfx950=../test-plans/sdpa-fwd/timings-gfx950.csv
```

The command writes `levels.csv` and `levels-summary.md` to the output folder.
It takes about 3 minutes. The same inputs always give the same `levels.csv`,
byte for byte.

## Check the CSV

`../test_sdpa_fwd_levels_regenerate.py` runs the command above and compares the
SHA-256 of `levels.csv` with `levels.csv.sha256`. CTest runs it as
`hipdnn_sdpa_fwd_levels_regenerate_python_tests` in the `comprehensive` tier.

When a change to the model, the timing files or the generator changes the CSV
on purpose, write the new SHA-256 to `levels.csv.sha256` in the same change.
