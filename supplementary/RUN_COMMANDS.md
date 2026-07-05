# Reproduction Commands

Run these commands from the repository root.

## 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The public UCI datasets are downloaded automatically into `data/` when the experiments are run.

## 2. Source-Level Diagnosis and Feature-Block Ablations

This command reproduces the source-level baseline matrix, HRA-Core/feature-block ablations, residual-index outputs, tuned reconstruction baselines, and flat-vs-hierarchical comparisons.

```powershell
python experiments\source_diagnosisun.py `
  --datasets synthetic uci_har air_quality `
  --seeds 0 1 2 3 4 `
  --severities 0.75 1.0 1.25 1.5 1.75 `
  --n-train-per-class 50 `
  --n-test-per-class 50 `
  --length 128 `
  --channels 8 `
  --max-base-windows 1400 `
  --ae-budgets 10 30 50 100 `
  --ae-patience 12 `
  --include-flat `
  --out runs\source_diagnosis_current
```

## 3. Supervised Affected-Channel Shortlisting

```powershell
python experiments\channel_shortlistingun.py `
  --datasets synthetic uci_har air_quality `
  --seeds 0 1 2 3 4 `
  --severities 0.75 1.0 1.25 1.5 1.75 `
  --n-train-per-class 50 `
  --n-test-per-class 50 `
  --length 128 `
  --channels 8 `
  --max-base-windows 1400 `
  --ae-budgets 100 `
  --ae-patience 12 `
  --out-dir runs\channel_shortlisting_current
```

## 4. Fault-Type Localization Breakdown

```powershell
python experimentsault_type_breakdownun.py `
  --datasets synthetic uci_har air_quality `
  --seeds 0 1 2 3 4 `
  --severities 0.75 1.0 1.25 1.5 1.75 `
  --n-train-per-class 50 `
  --n-test-per-class 50 `
  --length 128 `
  --channels 8 `
  --max-base-windows 1400 `
  --out runsault_type_breakdown_currentault_type_localization.csv `
  --raw-out runsault_type_breakdown_currentault_type_windows.csv
```

## 5. Regenerate Included Figures

```powershell
python supplementary\scripts\make_current_figures.py
```
