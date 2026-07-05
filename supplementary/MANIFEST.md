# Supplementary Artifact Manifest

This directory contains the reproducibility materials for the current HRA-SL submission package:

- HRA-SL as the overall diagnostic and affected-channel shortlisting framework.
- HRA-Core signature as the source-level core diagnostic signal.
- HRA+spectral as the main fused source-level configuration.
- Fixed HGB HRA-SL as the supervised affected-channel shortlisting localizer.

The artifact is intentionally limited to the configurations, tables, figures, and raw result files that support the current reported experiments.

## Directory Contents

- `current_tables/`: filtered CSV/TEX tables and paired-difference summaries matching the current reported tables and figures.
- `current_figures/`: Figure 1 and the current generated result figures in PDF/PNG form.
- `raw_current_results/`: filtered per-run/raw result files needed to verify the reported numbers.
- `configs/`: run configuration and split manifests for the current source-diagnosis, shortlisting, and fault-type analyses.
- `scripts/`: plotting script for regenerating the included figures from `current_tables/`.
- `RUN_COMMANDS.md`: reproduction commands using the cleaned GitHub source layout.
- `ENVIRONMENT.md`: environment snapshot from the machine used to assemble the artifact.

## Reported-Item Mapping

- Table 2: `current_tables/baseline_strengthened_summary_current.{csv,tex}`.
- Table 3: `current_tables/supervised_localizer_summary_current.{csv,tex}`.
- Table 4: `current_tables/ablation_feature_blocks_current.{csv,tex}`.
- Figure 1: `current_figures/figure1_hra_sl_workflow.png`.
- Figure 2: `current_figures/baseline_strengthened_macro_f1.{pdf,png}`.
- Figure 3: `current_figures/feature_block_ablation.{pdf,png}`.
- Figure 4: `current_figures/hierarchy_vs_flat.{pdf,png}`.
- Figure 5: `current_figures/fault_type_breakdown.{pdf,png}`.
