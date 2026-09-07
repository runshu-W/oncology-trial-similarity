#!/usr/bin/env bash
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round12
TAR=$ROOT/revision_round12_2026-09-07.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript/figures,scripts,simulation/gold_standard,results/tables,results/figures/final_v4_selective,docs,artifacts/round4/calibration}
cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/figures/FS1_crossfit_dist.pdf "$PKG/manuscript/figures/"
cp "$ROOT"/results/figures/final_v4_selective/FS1_crossfit_dist.pdf "$PKG/results/figures/final_v4_selective/"
cp "$ROOT"/scripts/build_round9_figures.py "$ROOT"/scripts/check_page_geometry.py "$ROOT"/scripts/verify_round12.py "$ROOT"/scripts/build_package_round12.sh "$PKG/scripts/"
cp "$ROOT"/simulation/gold_standard/round12_eval_sizes.py "$PKG/simulation/gold_standard/"
cp "$ROOT"/results/tables/round12_eval_sizes.json "$PKG/results/tables/"
cp "$ROOT"/docs/response_to_review_round12_2026-09-07.md "$PKG/docs/"
cp "$ROOT"/artifacts_round4/calibration/round12_eval_sizes.json "$PKG/artifacts/round4/calibration/"
( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"; md5sum "$TAR"
