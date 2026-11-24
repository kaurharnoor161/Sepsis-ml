# Sepsis-ml: early sepsis prediction on ICU data

Predicting sepsis hour by hour for ICU patients, using the public
PhysioNet/CinC Challenge 2019 data (hospital A and hospital B,
40,336 patients, 1,552,210 patient-hours, 7.27% of patients become septic).

Everything is split **by patient**, never by row, so no patient appears in
both training and testing. Test results are scored once, after thresholds
are fixed on validation.

## Results (test set, 6,051 patients)

| Model | What it is | AUROC | AUPRC |
|---|---|---|---|
| M0 | SIRS rule (>= 2 criteria) | 0.638 | 0.029 |
| M1 | Logistic regression | 0.787 | 0.098 |
| M2 | Decision tree | 0.786 | 0.093 |
| M3 | Random forest (50 core features) | 0.833 | 0.108 |
| M3b | Random forest (all 239 features) | **0.844** | **0.114** |
| M4 | Gaussian Naive Bayes | 0.757 | 0.082 |
| M5 | k-nearest neighbours | 0.793 | 0.083 |
| M6 | RBF SVM | 0.813 | 0.081 |
| M7 | Neural network (MLP, 64-32) | 0.819 | 0.110 |

Only about 1.8% of patient-hours are positive, so AUPRC values near 0.1 are
the realistic scale here (a random model would score about 0.018).
Bootstrap confidence intervals and paired differences are in
`results/` (step 13).

## Pipeline (one notebook per step, run in order)

| Notebook | What it does |
|---|---|
| 01_data_audit | Loads the official A/B files, checks them against the Kaggle CSV |
| 02_eda | Exploratory analysis, missingness, pre-onset trajectories |
| 03_patient_splits | 70/15/15 patient split, 5 CV folds, cross-hospital splits |
| 04_preprocessing | Clipping, scaling, missing-value features (fit on train only) |
| 05_feature_engineering | 239 causal features: rolling stats, counts, SIRS, extremes |
| 06_reduction | Feature selection (core set of 50), PCA, rate features |
| 07_regression | Predicts stay length from hour-7 information |
| 08_classification | M0-M3: SIRS, logistic regression, tree, random forest |
| 08b_classification_nb_knn_svm | M4-M6: Naive Bayes, KNN, SVM |
| 09_clustering | K-Means, Ward, DBSCAN on first 8 hours |
| 10_association_rules | Apriori rules with sepsis as the outcome |
| 11_ann | M7: neural network, ReLU vs tanh vs sigmoid |
| 12_ablations | Which feature groups matter, class imbalance, cross-hospital |
| 13_final_comparison | Final tables, bootstrap CIs, alert timing, calibration |

`src/sepsis_eval.py` holds the shared metric code (sensitivity, F2, AUROC,
AUPRC, alert timing, false alerts per 100 patient-days).
Each step also writes a text log to `results/stepNN_summary.txt`.

## Main findings

- Missingness itself is informative: how often labs are ordered predicts
  sepsis, and adding missingness features helps every model.
- Cross-hospital tests lose accuracy. Logistic regression drops the most
  (e.g. AUROC 0.80 -> 0.72 from A to B); random forest is steadier.
- Many alerts are "too early" compared with the official label time, and
  all models give many false alerts at useful sensitivity.

## How to run

1. Download the PhysioNet/CinC 2019 training data (sets A and B) and the
   Kaggle `Dataset.csv`, and place them in a `dataset/` folder at the
   repo root:
   `dataset/training_setA/`, `dataset/training_setB/`, `dataset/Dataset.csv`.
   (This folder is not in the repo.)
2. Install with [uv](https://docs.astral.sh/uv/): `uv sync`
3. Open the notebooks in `notebooks/` and run 01 to 13 in order.
   Each one finds the project root by looking for the `dataset/` folder.
   Generated files go to `data/processed/` and `results/`.

Requires Python 3.13 or newer.

## Known notes

- `ICULOS == hour + 1` fails by design in step 02: some patients enter the
  data after their first ICU hour. Step 03 confirms ICULOS is consistent
  with their first-hour offset.
- In step 12 one input check flags `PC1-PC3` missing from the parquet files.
  The PCA columns are not used by the ablation models.
- KNN and SVM are trained on a fixed 100,000-row subset and are not
  cross-validated; validation scores at a validation-tuned threshold are
  optimistic by design, which is why test is scored once at the end.
