#!/usr/bin/env bash
# Assemble the round-5 revision package (repo-relative layout) and tarball.
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round5
TAR=$ROOT/revision_round5_2026-08-31.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript/figures,pipeline,scripts,simulation/gold_standard,results/tables,results/figures/final_v4_selective,docs,artifacts/round4/calibration,artifacts/round4/stress,artifacts/round4/starvation}

cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"
for f in F1_headtohead_nll F7_pipeline_schematic; do
  cp "$ROOT/manuscript/figures/$f.pdf" "$PKG/manuscript/figures/"
done

cp "$ROOT"/pipeline/build_main_figures_selective.py "$PKG/pipeline/"
cp "$ROOT"/pipeline/build_pipeline_schematic.py "$PKG/pipeline/"

for f in equal_n_extra_round5.py influence_diag_round5.py \
         capped_table_rows_round5.py build_package_round5.sh; do
  cp "$ROOT/scripts/$f" "$PKG/scripts/"
done

for f in size_matched_round5.py mixing_robustness_round5.py \
         run_round5_stress.py; do
  cp "$ROOT/simulation/gold_standard/$f" "$PKG/simulation/gold_standard/"
done
cp "$ROOT/simulation/gold_standard/dgm.py" "$PKG/simulation/gold_standard/"

for f in gold_standard_size_matched.json gold_standard_mixing_robustness.json \
         gold_standard_round5_stress.json influence_diag.json \
         capped_table_rows.json equal_n_extra.json; do
  cp "$ROOT/results/tables/$f" "$PKG/results/tables/"
done

for f in F1_headtohead_nll F7_pipeline_schematic; do
  cp "$ROOT/results/figures/final_v4_selective/$f.pdf" "$PKG/results/figures/final_v4_selective/"
  cp "$ROOT/results/figures/final_v4_selective/$f.tif" "$PKG/results/figures/final_v4_selective/" 2>/dev/null || true
done

cp "$ROOT"/docs/response_to_review_round5_2026-08-31.md "$PKG/docs/"

cp "$ROOT"/artifacts_round4/calibration/size_matched_round5.json \
   "$ROOT"/artifacts_round4/calibration/mixing_robustness_round5.json \
   "$PKG/artifacts/round4/calibration/"
cp -R "$ROOT"/artifacts_round4/stress/. "$PKG/artifacts/round4/stress/"
cp "$ROOT"/artifacts_round4/starvation/equal_n_extra.json \
   "$PKG/artifacts/round4/starvation/"

( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"
md5sum "$TAR"
