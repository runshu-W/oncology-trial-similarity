#!/usr/bin/env bash
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round10
TAR=$ROOT/revision_round10_2026-09-05.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript/figures,scripts,pipeline,simulation/gold_standard,results/tables,results/figures/final_v4_selective,docs,artifacts/round4/calibration,artifacts/round4/stress}
cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"
for f in F1_headtohead_nll F2_forward_validation F3_calibration F12_triage FS1_crossfit_dist; do
  cp "$ROOT/manuscript/figures/$f.pdf" "$PKG/manuscript/figures/"
  cp "$ROOT/results/figures/final_v4_selective/$f.pdf" "$PKG/results/figures/final_v4_selective/"
  [ -f "$ROOT/results/figures/final_v4_selective/$f.tif" ] && cp "$ROOT/results/figures/final_v4_selective/$f.tif" "$PKG/results/figures/final_v4_selective/"
done
cp "$ROOT"/scripts/round10_triage_capped.py "$PKG/scripts/"
cp "$ROOT"/scripts/build_round9_figures.py "$PKG/scripts/"
cp "$ROOT"/scripts/verify_round10.py "$PKG/scripts/"
cp "$ROOT"/scripts/build_package_round10.sh "$PKG/scripts/"
cp "$ROOT"/pipeline/build_main_figures_selective.py "$PKG/pipeline/"
cp "$ROOT"/simulation/gold_standard/round10_randomized_crossfit.py "$PKG/simulation/gold_standard/"
cp "$ROOT"/results/tables/round10_randomized_crossfit.json "$ROOT"/results/tables/round10_triage_capped.json "$PKG/results/tables/"
cp "$ROOT"/docs/response_to_review_round10_2026-09-05.md "$PKG/docs/"
cp "$ROOT"/artifacts_round4/calibration/round10_randomized_crossfit.json "$PKG/artifacts/round4/calibration/"
cp "$ROOT"/artifacts_round4/stress/round10_triage_capped.json "$PKG/artifacts/round4/stress/"
( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"; md5sum "$TAR"
