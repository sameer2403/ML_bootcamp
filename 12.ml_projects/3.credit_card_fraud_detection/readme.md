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

## Layman

Imagine a security guard watching 578 people walk by for every 1 pickpocket. If the guard just shouts "everyone's fine!" all day, they're right 99.8% of the time — and completely useless. That's the trap with fraud data: a model can look 99.8% accurate while catching zero fraud.

So instead of asking "how often is the model right overall," this project asks "when the model says fraud, is it actually fraud, and does it catch most of the real fraud?" — that's precision and recall, and PR-AUC is the single number that summarizes the trade-off between them across every possible sensitivity setting.

The fix for the imbalance was to make the model pay more attention to the rare fraud cases during training (`class_weight="balanced"`), rather than let it get lazy and just predict "not fraud" every time.

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
|---|---|---|---|---|---|
| Train | 100.0% | 1.00 | 1.00 | 1.00 | 1.000 |
| Test | 99.95% | 0.96 | 0.76 | 0.85 | 0.868 |

- **Baseline (unweighted Decision Tree) for contrast:** Test PR-AUC was only **0.561**, with fraud F1 of 0.75 — the balanced Random Forest is a meaningfully stronger fraud detector, not just a marginal gain.
- **Read on train PR-AUC = 1.0:** the tree ensemble is overfitting on training data (expected for RF without depth constraints tight enough to fully generalize), but the test PR-AUC gap (1.0 → 0.868) is the number that matters, and it still comfortably beats the baseline and the other candidate models.

---

## Code

```python
# --- Preprocessing ---
df["Amount_log1p"] = np.log1p(df["Amount"])
X = df.drop(columns=["Class", "Amount"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# --- Model comparison (5-fold Stratified CV on PR-AUC) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    "dt_balanced": Pipeline([("model", DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"))]),
    "hgb_balanced": Pipeline([("model", HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"))]),
    "rf_balanced": Pipeline([("model", RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"))]),
}

for name, m in models.items():
    pr_auc_scores = []
    for tr_idx, te_idx in cv.split(X_train, y_train):
        X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]
        m.fit(X_tr, y_tr)
        pred_prob = m.predict_proba(X_te)[:, 1]
        pr_auc_scores.append(average_precision_score(y_te, pred_prob))
    print(name, "CV PR-AUC mean:", round(np.mean(pr_auc_scores), 4))

# --- Hyperparameter tuning (winning model: Random Forest) ---
rfc_pipeline = Pipeline([("model", RandomForestClassifier(
    random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"))])

param_dist = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_leaf": [1, 5, 10, 20],
    "model__max_features": ["sqrt", "log2", 0.5],
}

random_search = RandomizedSearchCV(
    estimator=rfc_pipeline, param_distributions=param_dist,
    n_iter=10, scoring="average_precision", cv=cv, verbose=True,
)
random_search.fit(X_train, y_train)

# --- Final model + inference ---
best_rfc = Pipeline([("model", RandomForestClassifier(
    n_estimators=300, min_samples_leaf=1, max_features="log2",
    max_depth=30, random_state=RANDOM_STATE, n_jobs=-1,
    class_weight="balanced_subsample"))])
best_rfc.fit(X_train, y_train)

def predict_class(input_features):
    input_df = pd.DataFrame([input_features], columns=X_train.columns)
    prediction = best_rfc.predict(input_df)
    print("Fraud Transaction detected ❌" if prediction[0] == 1 else "Non-Fraud Transaction ✅")
```

---

## Follow-up

### Approach
The notebook's own closing note flags the next lever: **tune the probability threshold** instead of relying on the default 0.5 cutoff. Since Random Forest outputs a fraud probability per transaction, sliding that decision threshold up or down directly trades recall against precision — useful because in fraud detection, missing a fraud (false negative) and wrongly blocking a legitimate transaction (false positive) usually carry very different real-world costs.

### Layman
Right now the model says "fraud" only if it's more than 50% confident. But a bank might prefer to say "fraud" at 30% confidence, and eat a few more false alarms, if catching more real fraud saves more money than an annoying card decline costs — or the opposite. Threshold tuning is dialing that confidence knob to match what actually matters to the business, without touching the model itself.

### Technical
Concretely, the next step would be to compute the full precision-recall curve on the test set (`sklearn.metrics.precision_recall_curve`), pick a threshold that hits a target recall (e.g. 90% of fraud caught) or a target precision, and re-derive the confusion matrix at that cutoff. Other extensions worth exploring: SMOTE/undersampling as an alternative to class-weighting, cost-sensitive learning with an explicit false-positive/false-negative cost matrix, and calibrating the RF's probabilities (`CalibratedClassifierCV`) since tree ensembles are known to produce poorly calibrated probability estimates.