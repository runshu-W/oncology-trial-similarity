#!/usr/bin/env bash
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round11
TAR=$ROOT/revision_round11_2026-09-06.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript,scripts,simulation/gold_standard,results/tables,docs,artifacts/round4/calibration}
cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/scripts/verify_round11.py "$ROOT"/scripts/build_package_round11.sh "$PKG/scripts/"
cp "$ROOT"/simulation/gold_standard/round11_inference_scope.py "$PKG/simulation/gold_standard/"
cp "$ROOT"/results/tables/round11_inference_scope.json "$PKG/results/tables/"
cp "$ROOT"/docs/response_to_review_round11_2026-09-06.md "$PKG/docs/"
cp "$ROOT"/artifacts_round4/calibration/round11_inference_scope.json "$PKG/artifacts/round4/calibration/"
( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"; md5sum "$TAR"
