#!/usr/bin/env bash
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round8
TAR=$ROOT/revision_round8_2026-08-31.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript/figures,scripts,simulation/gold_standard,results/tables,results/figures/final_v4_selective,docs,artifacts/round4/calibration,artifacts/round4/stress}
cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/figures/F13_primary.pdf "$ROOT"/manuscript/figures/F14_profiles.pdf "$PKG/manuscript/figures/"
cp "$ROOT"/scripts/build_round4_figures.py "$PKG/scripts/"
cp "$ROOT"/scripts/round8_direct_paired.py "$PKG/scripts/"
cp "$ROOT"/scripts/verify_round8.py "$PKG/scripts/"
cp "$ROOT"/scripts/build_package_round8.sh "$PKG/scripts/"
cp "$ROOT"/simulation/gold_standard/round8_crossfit_dist.py "$PKG/simulation/gold_standard/"
cp "$ROOT"/results/tables/round8_direct_paired.json "$ROOT"/results/tables/gold_standard_round8_crossfit_dist.json "$PKG/results/tables/"
cp "$ROOT"/results/figures/final_v4_selective/F13_primary.pdf "$ROOT"/results/figures/final_v4_selective/F14_profiles.pdf "$PKG/results/figures/final_v4_selective/"
cp "$ROOT"/docs/response_to_review_round8_2026-08-31.md "$PKG/docs/"
cp "$ROOT"/artifacts_round4/calibration/round8_crossfit_dist.json "$PKG/artifacts/round4/calibration/"
cp "$ROOT"/artifacts_round4/stress/round8_direct_paired.json "$PKG/artifacts/round4/stress/"
( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"; md5sum "$TAR"
