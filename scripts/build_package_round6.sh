#!/usr/bin/env bash
# Assemble the round-6 revision package (repo-relative layout) and tarball.
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round6
TAR=$ROOT/revision_round6_2026-08-31.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript/figures,scripts,simulation/gold_standard,results/tables,docs,artifacts/round4/calibration,artifacts/round4/stress}

cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"

cp "$ROOT"/scripts/round6_block_capped.py "$PKG/scripts/"
cp "$ROOT"/scripts/build_package_round6.sh "$PKG/scripts/"
cp "$ROOT"/simulation/gold_standard/round6_analysis.py "$PKG/simulation/gold_standard/"

cp "$ROOT"/results/tables/gold_standard_round6_analysis.json \
   "$ROOT"/results/tables/round6_block_capped.json "$PKG/results/tables/"

cp "$ROOT"/docs/response_to_review_round6_2026-08-31.md "$PKG/docs/"

cp "$ROOT"/artifacts_round4/calibration/round6_analysis.json "$PKG/artifacts/round4/calibration/"
cp "$ROOT"/artifacts_round4/stress/round6_block_capped.json "$PKG/artifacts/round4/stress/"

( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"
md5sum "$TAR"
