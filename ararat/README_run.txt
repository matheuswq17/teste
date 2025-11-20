Run book for the radiomics pipeline (non-deep learning) for prostate cancer risk classification (GGG ≥3 vs ≤2).

Environment and reproducibility
- Python version: pin to 3.10+
- Key packages: numpy, pandas, scikit-learn, lightgbm, shap, SimpleITK, pydicom, pyradiomics, matplotlib, seaborn
- Seeds: use a fixed SEED constant (e.g., 42) everywhere random numbers are created
- Version logging: write package versions and commit hash into exports/report.md when available

Entry points
1) ararat/src/extract_features.py
   - Reads DICOM/NIfTI images and segmentations, resamples to 1x1x1 mm, normalizes per fold, extracts classic radiomics features (original_* only) with PyRadiomics.
   - Outputs a feature CSV with PatientID, LesionID, Target, and feature columns.

2) ararat/src/train_validate.py
   - Loads feature CSV, builds stratified group folds by PatientID to avoid leakage, performs feature selection per fold, trains Logistic Regression, Random Forest, and LightGBM.
   - Generates out-of-fold predictions, selects the best model by F1, calibrates probabilities using OOF data, and writes comparison tables and OOF predictions to exports.

3) ararat/src/evaluate_holdout.py
   - Retrains the best model on train+val, applies calibrated probabilities to the untouched holdout, computes ROC/PR with patient-level bootstrapped CIs, and saves plots/CSVs.

4) ararat/src/explain_shap.py
   - Uses SHAP TreeExplainer on the best model, generates beeswarm/decision/dependence plots, and writes them into the export folder.

Data expectations
- PROSTATEx mpMRI (T2/ADC) DICOM with accompanying DICOM-SEG or NIfTI masks.
- Manifest CSV should map PatientID and LesionID to image and mask paths.

Execution order
1) python ararat/src/extract_features.py --config ararat/configs/config.yaml
2) python ararat/src/train_validate.py --config ararat/configs/config.yaml
3) python ararat/src/evaluate_holdout.py --config ararat/configs/config.yaml
4) python ararat/src/explain_shap.py --config ararat/configs/config.yaml

Outputs
- exports/compete_no_overfit_FINAL_YYYYMMDD_HHMM contains CSVs, figures, and markdown reports required for the study.
