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

step "1/6 normalise names and addresses";        $PYTHON -m src.normalize
step "2/6 blocking (train split)";                $PYTHON -m src.block --split train --force
step "3/6 blocking (test split)";                 $PYTHON -m src.block --split test --force
step "4/6 train LightGBM";                        $PYTHON -m src.train --rebuild
step "5/6 tune decision rule";                    $PYTHON -m src.tune
step "6/6 predict test and write outputs";        $PYTHON -m src.predict

echo
echo "Done. Outputs are in ${ER_OUTPUT_DIR:-../../output}/"
