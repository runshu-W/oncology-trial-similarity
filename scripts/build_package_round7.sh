#!/usr/bin/env bash
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round7
TAR=$ROOT/revision_round7_2026-08-31.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript/figures,scripts,simulation/gold_standard,results/tables,results/figures/final_v4_selective,docs,artifacts/round4/calibration,artifacts/round4/stress}
cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/figures/F13_primary.pdf "$ROOT"/manuscript/figures/F14_profiles.pdf "$PKG/manuscript/figures/"
cp "$ROOT"/scripts/build_round4_figures.py "$PKG/scripts/"
cp "$ROOT"/scripts/build_package_round7.sh "$PKG/scripts/"
cp "$ROOT"/simulation/gold_standard/round7_crossfit.py "$PKG/simulation/gold_standard/"
cp "$ROOT"/results/tables/gold_standard_round7_crossfit.json "$ROOT"/results/tables/round7_block_rules.json "$PKG/results/tables/"
cp "$ROOT"/results/figures/final_v4_selective/F13_primary.pdf "$ROOT"/results/figures/final_v4_selective/F14_profiles.pdf "$PKG/results/figures/final_v4_selective/"
cp "$ROOT"/docs/response_to_review_round7_2026-08-31.md "$PKG/docs/"
cp "$ROOT"/artifacts_round4/calibration/round7_crossfit.json "$PKG/artifacts/round4/calibration/"
cp "$ROOT"/artifacts_round4/stress/round7_block_rules.json "$PKG/artifacts/round4/stress/"
( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"; md5sum "$TAR"
