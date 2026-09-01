#!/usr/bin/env bash
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round9
TAR=$ROOT/revision_round9_2026-08-31.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript/figures,scripts,pipeline,simulation/gold_standard,results/tables,results/figures/final_v4_selective,docs,artifacts/round4/calibration,artifacts/round4/stress}
cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"
for f in F1_headtohead_nll F2_forward_validation F3_calibration F12_triage FS1_crossfit_dist; do
  cp "$ROOT/manuscript/figures/$f.pdf" "$PKG/manuscript/figures/"
  cp "$ROOT/results/figures/final_v4_selective/$f.pdf" "$PKG/results/figures/final_v4_selective/"
  [ -f "$ROOT/results/figures/final_v4_selective/$f.tif" ] && cp "$ROOT/results/figures/final_v4_selective/$f.tif" "$PKG/results/figures/final_v4_selective/"
done
cp "$ROOT"/scripts/round9_tail_robustness.py "$PKG/scripts/"
cp "$ROOT"/scripts/build_round9_figures.py "$PKG/scripts/"
cp "$ROOT"/scripts/verify_round9.py "$PKG/scripts/"
cp "$ROOT"/scripts/build_package_round9.sh "$PKG/scripts/"
cp "$ROOT"/pipeline/build_main_figures_selective.py "$PKG/pipeline/"
cp "$ROOT"/simulation/gold_standard/round9_crossfit_diag.py "$PKG/simulation/gold_standard/"
cp "$ROOT"/results/tables/round9_tail_robustness.json "$ROOT"/results/tables/round9_crossfit_diag.json "$PKG/results/tables/"
cp "$ROOT"/docs/response_to_review_round9_2026-08-31.md "$PKG/docs/"
cp "$ROOT"/artifacts_round4/calibration/round9_crossfit_diag.json "$PKG/artifacts/round4/calibration/"
cp "$ROOT"/artifacts_round4/stress/round9_tail_robustness.json "$ROOT"/artifacts_round4/stress/round9_rows.json "$PKG/artifacts/round4/stress/"
( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"; md5sum "$TAR"
