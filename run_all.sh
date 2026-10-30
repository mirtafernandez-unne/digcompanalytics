#!/usr/bin/env bash
# Runs the complete DIGCOMPAnalytics evaluation.
# Usage: ./run_all.sh [LED_LOG.csv] [ITAA_LOG.csv]
# Without arguments, synthetic logs are generated and analyzed.
# Set DCA_N_PERT / DCA_N_BOOT to reduce the number of perturbations / bootstrap resamples for a quick run.
set -euo pipefail
cd "$(dirname "$0")"
LED=${1:-data/synthetic/led_synthetic.csv}
ITAA=${2:-data/synthetic/itaa_synthetic.csv}
if [ ! -f "$LED" ] || [ ! -f "$ITAA" ]; then python scripts/make_synthetic.py; fi
OUT=results
mkdir -p "$OUT"
python src/analysis.py "$LED" "$ITAA" config/mapping_digcomp22.csv "$OUT"
python src/transfer_operational.py "$OUT" "$OUT/transfer_operational.json"
python src/robustness.py "$LED" "$ITAA" config/mapping_digcomp22.csv "$OUT/robustness.json"
python src/scenario_navigation.py "$LED" "$ITAA" config/mapping_digcomp22.csv "$OUT/scenario_navigation.json"
echo "Results written to $OUT/"
