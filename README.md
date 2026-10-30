# Endpoint label revision: saved-result reproduction

This repository provides scripts and the derived inputs required to recompute selected results in the manuscript **Endpoint label revisions and the reliability of battery lifetime model comparisons**. The code checks the archived labels and model predictions; it does not reprocess raw battery curves, train models, establish the physically correct endpoint, or independently validate the conclusions.

## Run

Use Python 3.12 with NumPy 2.3.5 and pandas 3.0.1. From the repository root:

```sh
python -m pip install -r requirements.txt
python replay/verify_replay.py
python budget_extension/verify_budgets.py
```

The first command verifies input hashes and recalculates 18 expected outputs concerning label-generation branches, the original 23-seed ratios, the fixed-XGBoost-initialisation check, and the original versus doubled deep-model budgets. The second recomputes five-metric rankings and prediction-movement ratios from the saved predictions for the original, doubled, 38/40-epoch, and 74/78-epoch budgets; it also checks saved validation curves and the selected epochs. Both programs write result files inside their own directories. These checks use existing fitted-model predictions and the same previously examined test set; comparison counts are not independent replications.

Within-budget A/B winner agreement is 115/115 for the original 23 seeds; for the matched 10 seeds it is 50/50, 47/50, 43/50 and 38/50 at the original, doubled, 38/40 and 74/78 budgets, respectively. The 80-epoch search followed inspection of the 40-epoch validation curves; 78 triggered the predefined boundary warning. Selection used one internal split, and no independent test-set validation or convergence is established.

## Scope and data

The repository contains derived label tables, recorded predictions and validation curves, and study-specific analysis scripts. It does not include raw capacity trajectories, processed upstream battery objects, pretrained weights, or third-party model source files. The conformal and all figure-generation workflows are outside the replay in this repository. Full reproduction of training and preprocessing requires separately obtained upstream materials and the documented software environment; the historical Contract A generator and exact early holdout-selection script were not recovered. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `replay/README.md` for input lineage and limitations. The historical prediction filenames use the word `blind`; the test set has been repeatedly examined and is not an independent blind test here.

Tan R, Hong W, Tang J, et al. BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction. KDD 2025. https://doi.org/10.1145/3711896.3737372. Processed BatteryLife v12: https://zenodo.org/records/21149533. Generator revision: `3b17bb53202e10711deae0448c91f84144bb3dd9`.
