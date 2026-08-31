#!/usr/bin/env bash
# Assemble the round-4 revision package (repo-relative layout) and tarball.
set -euo pipefail
ROOT=~/work/osim
PKG=$ROOT/package_round4
TAR=$ROOT/revision_round4_2026-08-31.tar.gz
rm -rf "$PKG"
mkdir -p "$PKG"/{manuscript/figures,pipeline,scripts,simulation/gold_standard,results/tables,results/figures/final_v4_selective,docs,artifacts/round4}

cp "$ROOT"/manuscript/manuscript_full_draft_pharm_stats.{tex,pdf} "$PKG/manuscript/"
cp "$ROOT"/manuscript/supplement.{tex,pdf} "$PKG/manuscript/"
for f in F1_headtohead_nll F2_forward_validation F3_calibration F12_triage F13_primary F14_profiles; do
  cp "$ROOT/manuscript/figures/$f.pdf" "$PKG/manuscript/figures/"
done

cp "$ROOT"/pipeline/build_main_figures_selective.py "$PKG/pipeline/"

for f in capped_real_data.py starvation_controls.py drift_distribution.py \
         audit_sample_round4.py audit_score_round4.py build_round4_figures.py \
         build_package_round4.sh; do
  cp "$ROOT/scripts/$f" "$PKG/scripts/"
done

for f in run_round4_capsweep.py run_round4_negcells.py run_round4_validation.py \
         run_round4_ec_boot.py calibrate_round4.py validate_round4.py \
         n0_strata_round4.py; do
  cp "$ROOT/simulation/gold_standard/$f" "$PKG/simulation/gold_standard/"
done

for f in gold_standard_mixture_calibration.json gold_standard_validation_report.json \
         gold_standard_n0_strata_selection.json gold_standard_n0_strata_validation.json \
         gold_standard_ec_boot.json drift_distribution.json \
         real_capped_summary.json starvation_summary.json; do
  cp "$ROOT/results/tables/$f" "$PKG/results/tables/"
done
if [ -f "$ROOT/results/tables/oos_audit_results.json" ]; then
  cp "$ROOT/results/tables/oos_audit_results.json" "$PKG/results/tables/"
fi

for f in F1_headtohead_nll F2_forward_validation F3_calibration F12_triage; do
  cp "$ROOT/results/figures/final_v4_selective/$f.pdf" "$PKG/results/figures/final_v4_selective/"
  cp "$ROOT/results/figures/final_v4_selective/$f.tif" "$PKG/results/figures/final_v4_selective/" 2>/dev/null || true
done
cp "$ROOT/results/figures/final_v4_selective/F13_primary.pdf" \
   "$ROOT/results/figures/final_v4_selective/F14_profiles.pdf" \
   "$PKG/results/figures/final_v4_selective/"

cp "$ROOT"/docs/response_to_review_round4_2026-08-31.md "$PKG/docs/"
cp "$ROOT"/docs/endpoint_audit_sample_round4.csv "$PKG/docs/"
cp "$ROOT"/docs/endpoint_audit_README_round4.md "$PKG/docs/"
if [ -f "$ROOT/docs/endpoint_audit_sample_round4_completed.csv" ]; then
  cp "$ROOT/docs/endpoint_audit_sample_round4_completed.csv" "$PKG/docs/"
fi

cp -R "$ROOT"/artifacts_round4/. "$PKG/artifacts/round4/"

( cd "$PKG" && tar czf "$TAR" . )
echo "files: $(find "$PKG" -type f | wc -l)"
du -sh "$TAR"
md5sum "$TAR"
