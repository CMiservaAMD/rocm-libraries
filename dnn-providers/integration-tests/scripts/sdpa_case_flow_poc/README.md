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

## From the CSV to bundles

This section is the whole pipeline: model, then `levels.csv`, then graphs, then
bundles. With it you can make the `SdpaFwdPlan` bundle set again from the files
in the repository. The design behind each step is in
[SDPA Forward Test Plan](../../docs/test-plans/sdpa-fwd.md), sections 8 to 10.

The `SdpaFwdPlan` bundles of the pull request that adds them came from exactly
this process: the committed model files, seed 2893, a `levels.csv` with the
SHA-256 in `levels.csv.sha256`, and the default selection parameters of
`select_rows.py`. See [Known difference](#known-difference) for the one value
that a rebuild gives differently.

### Where the tools are

| Folder | What it holds |
|---|---|
| `scripts/sdpa_case_flow_poc/` | The generator: model to `levels.csv` (`levels.py`), and the class-to-value file (`python3 -m sdpa_case_flow_poc`). |
| `scripts/sdpa_case_flow_poc/bundles/` | The bundle tools: `select_rows.py`, `topology.py`, `compare_exports.py`, `make_bundles.py`, `annotate_csv.py`, and for the GPU steps `analyse.py`, `build_skips.py`, `analyse_refcheck.py`. Run them as scripts. |
| `src/sdpa-case-flow-poc/` | `CaseGraph.cpp`, which makes the hipDNN graph of a row, and a discovery driver (`hipdnn_sdpa_case_flow_discovery`). |
| `src/sdpa-case-flow-runner/` | `hipdnn_sdpa_case_flow_runner`: exports graphs, and on a GPU runs an engine and both references per row. |
| `migration-scripts/import_graph.py` | The suite's import tool. Its opt-in grouped mode writes the sweeps. |

Every input and output of the bundle tools is a command-line argument. Defaults
are relative to the repository. Run `<tool> --help` for the full list.

### Steps

Only the steps marked GPU need a GPU. Exporting graphs needs a build of the
runner, but no GPU. The times are what the steps took for the 9,883 bundle rows.

| Step | Tool | GPU | Output | Time |
|---|---|---|---|---|
| 1 | `sdpa_case_flow_poc.levels` | no | `levels.csv`, checked against `levels.csv.sha256` | about 3 min |
| 2 | `python3 -m sdpa_case_flow_poc` | no | `class-to-value.json` | seconds |
| 3 | `bundles/select_rows.py` | no | `selected.csv`, `exclusions.csv`, `summary.json` | under a minute |
| 4 | CMake | no | `hipdnn_sdpa_case_flow_runner` | depends on the host |
| 5 | runner `--export-graphs`, `bundles/compare_exports.py` | no | one graph and one metadata file per row, per arch | about 1 min per arch |
| 6 | `bundles/make_bundles.py` | no | the `SdpaFwdPlan` sweeps, `placements.csv` | 10 to 60 min; one import process per graph |
| 7 | checks, `bundles/annotate_csv.py` | no | check results, `cases.csv` | minutes |
| 8 | `hipdnn_integration_tests`, `bundles/analyse.py`, `bundles/build_skips.py` | GPU | test results, `[[test_skips]]` entries | minutes per tier and arch |
| 9 | runner, `bundles/analyse_refcheck.py` | GPU | reference acceptance and CPU-vs-GPU agreement per row | about 6 GPU hours per arch |

The commands below use these names:

```bash
REPO=<repository root>
IT=$REPO/dnn-providers/integration-tests
TOOLS=$IT/scripts/sdpa_case_flow_poc/bundles
OUT=<an empty work folder>
BUILD=<CMake build folder>
```

### 1. The CSV

```bash
cd $IT/scripts
python3 -m sdpa_case_flow_poc.levels \
    --model-dir ../test-plans/sdpa-fwd --out-dir $OUT/levels --seed 2893 \
    --timings gfx942=../test-plans/sdpa-fwd/timings-gfx942.csv \
    --timings gfx950=../test-plans/sdpa-fwd/timings-gfx950.csv
sha256sum -c <(sed "s|  levels.csv|  $OUT/levels/levels.csv|" ../test-plans/sdpa-fwd/levels.csv.sha256)
```

The check must print `OK`. If it does not, the model files, the timing files
or the generator changed, and the rest of the pipeline gives a different set.

### 2. The class-to-value file

The selection step and the runner read the class-to-value table of the model as
a JSON file.

```bash
cd $IT/scripts
python3 -m sdpa_case_flow_poc --model-dir ../test-plans/sdpa-fwd \
    --out-dir $OUT/discovery --modes "" --skip-second-strength
```

This writes `$OUT/discovery/class-to-value.json` and `manifest.yaml`. With
`--modes` left at its default, the command also writes the one-at-a-time and
pairwise sets that section 6 of the design document describes.

### 3. Select the rows that a bundle can express

```bash
python3 $TOOLS/select_rows.py --input $OUT/levels/levels.csv \
    --class-values $OUT/discovery/class-to-value.json --out-dir $OUT/select
```

`selected.csv` holds the selected rows, line for line as in `levels.csv`.
`exclusions.csv` gives every other row with its reasons (section 9.1 of the
design document). The defaults are: all four tiers, minimum sweep size 1, the
zero-stride filter on, the descale filter (`--descale-non-fp8-filter`) off.
With these defaults, and an input whose SHA-256 is the one in
`levels.csv.sha256`, the script asserts the counts it records: 9,883 rows in
293 sweeps. `--no-zero-stride-filter` selects the 6 zero-stride rows too; the
data-SDK change they need (PR #14709) is now on the develop branch.

`topology.py --input … --class-values … --out topologies.csv` checks that the
topology names and the tensor signatures map one to one, and lists them.

### 4. Build the runner

The runner is part of the integration-test project and needs hipDNN built with
SDPA support. Two options, both off by default, add the PoC targets:

| Option | Target |
|---|---|
| `HIPDNN_IT_ENABLE_SDPA_CASE_FLOW_RUNNER` | `hipdnn_sdpa_case_flow_runner` |
| `HIPDNN_IT_ENABLE_SDPA_CASE_FLOW_DISCOVERY` | `hipdnn_sdpa_case_flow_discovery` |

For example, with the repository superbuild:

```bash
cmake -S $REPO -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
    --toolchain $REPO/projects/hipdnn/cmake/ClangToolChain.cmake -DROCM_CMAKE_PATH=<ROCm root> \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hip-kernel-provider;hipdnn-integration-tests" \
    -DGPU_TARGETS="gfx942;gfx950" -DHIPDNN_ENABLE_SDPA=ON \
    -DHIPDNN_IT_ENABLE_SDPA_CASE_FLOW_RUNNER=ON -DHIPDNN_IT_ENABLE_SDPA_CASE_FLOW_DISCOVERY=ON
cmake --build $BUILD --target hipdnn_sdpa_case_flow_runner hipdnn_sdpa_case_flow_discovery
```

The steps that run engines (8 and 9) also need the engine plugin
(`libhip_kernel_provider.so`) and `hipdnn_integration_tests`, so build those
targets too, or build everything.

### 5. Export the graphs

The export loads no plugin, creates no handle and runs nothing on a GPU. For
each row it writes `<case_key>/<case_key>.json` (the graph) and
`<case_key>/<case_key>.meta.json` (the bundle metadata), plus one JSON line per
row with its outcome.

```bash
SHA=$(cut -d' ' -f1 $IT/test-plans/sdpa-fwd/levels.csv.sha256)
for ARCH in gfx942 gfx950; do
    $BUILD/bin/hipdnn_sdpa_case_flow_runner --input $OUT/select/selected.csv \
        --output $OUT/export-$ARCH.jsonl --class-values $OUT/discovery/class-to-value.json \
        --export-graphs $OUT/export-$ARCH --arch $ARCH --source-sha256 $SHA
done
python3 $TOOLS/compare_exports.py $OUT/export-gfx942 $OUT/export-gfx950 \
    --report $OUT/compare-exports.json --outcomes $OUT/export-gfx942.jsonl $OUT/export-gfx950.jsonl
```

Every selected row must have the outcome `exported`. Rows that do not are
excluded on a second run of step 3 with `--export-outcomes <jsonl>`. The two
exports must be equal: no selected row depends on the arch (section 9.2).

### 6. Make the bundles

`make_bundles.py` runs `import_graph.py` in grouped mode once per graph, sorted
by tier, topology and `case_key`, into an empty work tree. Each run uses
`--op-dir SdpaFwdPlan --topology <name> --template-attributes
--template-alignment --skeleton-value-form --only-op-dir --require-marked-sweep
--case-id <dtype_q>_<case_key> --strict`.

```bash
python3 $TOOLS/make_bundles.py --selected $OUT/select/selected.csv \
    --class-values $OUT/discovery/class-to-value.json --export-dir $OUT/export-gfx942 \
    --work-dir $OUT/bundles --out-dir $OUT/make-bundles
```

The work tree gets `<tier>/SdpaFwdPlan/<Topology>/` with `graph.template.json`,
`sweep.json` (compact layout) and an empty `support.json`. `--install-into
$IT/integration-test-bundles` then copies the four op folders into the
suite's tree; it refuses to overwrite a folder that exists. `--sample N`,
`--only-tier`, `--only-topology` and `--limit` import a subset, which is
useful to try a change quickly.

`placements.csv` maps each `case_key` to its tier, topology, case id, bundle
path and GTest name. `report.json` has the import times and the sweep sizes.

### 7. Verify

`make_bundles.py` checks the tree before it reports `OK`: each topology is one
sweep per tier, no sweep holds two topologies, the op folder holds no other
sweep, and every case expands back to its exported graph. `import_graph.py
--strict` also checks each case when it writes it.

Then install the op folders into the suite's tree (unless `--install-into`
did), run the suite's checks, and write the annotated CSV:

```bash
for T in quick standard comprehensive full; do
    cp -r $OUT/bundles/$T/SdpaFwdPlan $IT/integration-test-bundles/$T/
done
python3 $IT/reference-data-scripts/verify_golden_bundles.py \
    $IT/integration-test-bundles/{quick,standard,comprehensive,full}/SdpaFwdPlan
python3 $IT/scripts/verify_support_claims.py
python3 $IT/scripts/support_features/feature_matrix.py --check
python3 $TOOLS/annotate_csv.py --input $OUT/levels/levels.csv \
    --class-values $OUT/discovery/class-to-value.json --selection-dir $OUT/select \
    --placements $OUT/make-bundles/placements.csv --out $OUT/cases.csv
```

`verify_support_claims.py` and `feature_matrix.py` read the suite's tree.
The committed `SUPPORT_MATRIX_OVERVIEW.md` on this branch
already counts the `SdpaFwdPlan` bundles, so `--check` fails until the set is
installed and passes once it is. If your set differs, regenerate the overview
with `feature_matrix.py`. `cases.csv` is `levels.csv` with five columns added
(section 9.4 of the design document).

To compare a rebuilt set with a committed one, use `diff -r` on the op folders;
see [Known difference](#known-difference).

### 8. Run the engines (GPU)

Run `hipdnn_integration_tests` with the ASM_SDPA_ENGINE lane's arguments, one
tier at a time. The lane's exact arguments are printed by
`ctest --test-dir $BUILD/dnn-providers/hip-kernel-provider -R '^hip-kernel-provider-asm-sdpa-external-integration_quick_suite$' -N -V`.
For example, on a gfx942 GPU:

```bash
RUNS=$OUT/results/validate/gfx942-job1
mkdir -p $RUNS
$BUILD/bin/hipdnn_integration_tests \
    --test-article $BUILD/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
    --test-engine ASM_SDPA_ENGINE \
    --test-config $REPO/dnn-providers/hip-kernel-provider/config/ASM_SDPA_ENGINE.toml \
    --golden-data-dir $OUT/bundles \
    --gtest_filter='quick_SdpaFwdPlan_*' --gtest_output=xml:$RUNS/quick.xml > $RUNS/quick.log 2>&1
```

Repeat for `standard`, `comprehensive` and `full`, and on gfx950 into
`validate/gfx950-job1`. To see the failures that the skips hide, run with a copy
of the TOML that ends before the line `# -- SdpaFwdPlan known engine defects --`.
A failing case can be run again with `HIPDNN_TEST_HOST_INPUT_FILL=1` into
`<tier>-hostfill-<n>.xml`, to see whether it depends on the input fill.

Then:

```bash
python3 $TOOLS/analyse.py --results $OUT/results --levels $OUT/levels/levels.csv \
    --placements $OUT/make-bundles/placements.csv --bundle-dir $OUT/bundles --part a
python3 $TOOLS/build_skips.py --validate $OUT/results/validate \
    --levels $OUT/levels/levels.csv --placements $OUT/make-bundles/placements.csv \
    --bundle-dir $OUT/bundles --out-dir $OUT/skips --dry-run
```

`analyse.py` counts passed, failed, declined, skipped and unverifiable cases per
arch and tier, and maps each failure to a known defect tag. A failure with no
tag is reported as `UNMAPPED`; trace it before you add a skip (section 10.2 of
the design document). `build_skips.py` without `--dry-run` writes the
`[[test_skips]]` entries at the end of the engine TOML, after the marker line,
and replaces an earlier section of its own. `analyse.py --skips <TOML>` then
checks that every failure is skipped under the right reason and that no passing
case is skipped.

The runner can also run every row against one engine, with
`--plugin-path <plugin> --engine <name> --expect-arch <arch>`, into
`results/runner/<arch>-job<id>/<engine>/results.jsonl`; `analyse.py --part b`
summarises those runs.

### 9. Reference check (GPU)

The suite checks each output against the hipDNN GPU and CPU SDPA references.
The reference check runs both references on every selected row and compares
them. It pins an engine that declines every SDPA graph, so no engine runs:

```bash
$BUILD/bin/hipdnn_sdpa_case_flow_runner --input $OUT/select/selected.csv \
    --output $OUT/refcheck/gfx942/rows.jsonl --class-values $OUT/discovery/class-to-value.json \
    --plugin-path $BUILD/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
    --engine HIP_MLOPS_ENGINE --expect-arch gfx942 \
    --cpu-ref-max-mflop 1e12 --max-tensor-mb 20000 --warmup 0 --iterations 1
python3 $TOOLS/analyse_refcheck.py --rows $OUT/select/selected.csv \
    --placements $OUT/make-bundles/placements.csv --results "$OUT/refcheck/*/rows.jsonl"
```

The CPU reference makes this slow: about 6 hours per arch on one GPU node.
`--start N --limit M` split the rows into shards that run in parallel.
`analyse_refcheck.py` reports, per arch and tier, the rows with no result, each
reference refusal, each CPU-vs-GPU disagreement, and the share of fp8 outputs
that differ by one representable value. For the pull request's set, both
references accepted and agreed on every row of quick, standard and
comprehensive; in full, both declined the same 60 rows on both arches.

### Known difference

Each case's metadata has `generator_version`: the first 12 hex digits of a
SHA-256 over the runner's `Main.cpp` and `CaseGraph.cpp`/`CaseGraph.hpp`. The
pull request's bundles carry `fadf8dc7c3ba`. On this branch the runner has a
later change to its reference comparison (element statistics for the
reference check), it accepts any output folder, and the three files are
formatted with clang-format. None of these changes the exported graphs, but
they change the sources, so a rebuilt set carries a different
`generator_version`. Every other
byte of the rebuilt set is expected to equal the committed set.
