# — Project : Credit Card Fraud Detection

**Dataset:** `creditcard.csv` (Kaggle — `mlg-ulb/creditcardfraud`) — 284,807 transactions, 31 columns, highly imbalanced (492 fraud vs 284,315 non-fraud → ~1 fraud per 578 non-fraud transactions).

---

## Approach

Build a classifier that flags fraudulent card transactions in a dataset where fraud is only 0.17% of records — meaning accuracy is a useless metric and the real fight is against class imbalance.

Workflow:
1. **EDA** — checked shape, nulls, duplicates (1,081 found, kept), class distribution, and the `Amount`/`Time` distributions. Confirmed near-zero inter-feature correlation (expected, since `V1–V28` are PCA components).
2. **Preprocessing** — log-transformed `Amount` (`Amount_log1p`) to tame its heavy right skew, dropped raw `Amount`, stratified 80/20 train-test split.
3. **Baseline** — plain Decision Tree (no imbalance handling) to establish a "do-nothing" reference point.
4. **Model selection** — compared Decision Tree, HistGradientBoosting, and Random Forest, all with `class_weight="balanced"`, using 5-fold Stratified CV scored on **PR-AUC (average precision)** — not accuracy or ROC-AUC — since PR-AUC is the metric that actually rewards catching rare-positive cases.
5. **Tuning** — `RandomizedSearchCV` on the winning model (Random Forest), scored on PR-AUC.
6. **Evaluation** — confusion matrix + classification report + PR-AUC on train and test.
7. **Inference** — a `predict_class()` function wrapping the final pipeline.

---

## Technical

- **Features:** `Time`, `Amount_log1p`, and 28 PCA components (`V1`–`V28`); raw `Amount` dropped in favor of its log1p transform for numerical stability.
- **Imbalance handling:** class-weighting (`balanced` / `balanced_subsample`) rather than resampling (no SMOTE/undersampling used here).
- **Model comparison (5-fold Stratified CV, PR-AUC):**

| Model                                   | CV PR-AUC (mean) |
|-----------------------------------------|-----------------------------------------|
| Decision Tree (balanced)                | 0.537 |
| HistGradientBoosting (balanced)         | 0.746 |
| **Random Forest (balanced_subsample)** | **0.837** ✅ |

- **Hyperparameter tuning** (`RandomizedSearchCV`, 10 iterations, PR-AUC scoring): best CV PR-AUC = **0.848**, with `n_estimators=500`, `max_depth=30`, `min_samples_leaf=1`, `max_features="log2"`.
- **Final model — Random Forest, tuned:**

| Split | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 | PR-AUC |
|-------|----------|-----------------|--------------|----------|----------|
| Train | 100.0%   | 1.00            | 1.00         | 1.00     | 1.000 |
| Test  | 99.95%   | 0.96            | 0.76         | 0.85     | 0.868 |

- **Baseline (unweighted Decision Tree) for contrast:** Test PR-AUC was only **0.561**, with fraud F1 of 0.75 — the balanced Random Forest is a meaningfully stronger fraud detector, not just a marginal gain.
- **Read on train PR-AUC = 1.0:** the tree ensemble is overfitting on training data (expected for RF without depth constraints tight enough to fully generalize), but the test PR-AUC gap (1.0 → 0.868) is the number that matters, and it still comfortably beats the baseline and the other candidate models.

---
