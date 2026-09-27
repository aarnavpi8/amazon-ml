#!/usr/bin/env bash
# End-to-end pipeline: challenge TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
#
# Run from anywhere (with the virtual environment activated):
#     bash code/business_entity_resolution/run_pipeline.sh
# Paths can be overridden with ER_DATA_DIR / ER_WORK_DIR / ER_OUTPUT_DIR (see src/config.py),
# and the interpreter with PYTHON (default: python).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

step() { echo; echo "=== $1"; }

step "1/7 normalise names and addresses";        $PYTHON -m src.normalize
step "2/7 infer missing states from city names";  $PYTHON -m src.infer_state
step "3/7 blocking (train split)";                $PYTHON -m src.block --split train --force
step "4/7 blocking (test split)";                 $PYTHON -m src.block --split test --force
step "5/7 train LightGBM";                        $PYTHON -m src.train --rebuild
step "6/7 tune decision rule";                    $PYTHON -m src.tune
step "7/7 predict test and write outputs";        $PYTHON -m src.predict

echo
echo "Done. Outputs are in ${ER_OUTPUT_DIR:-../../output}/"
