# 1 — Saving & Loading Trained Models (`joblib` vs `pickle`)

**Use case:** Pima Indians Diabetes dataset (`diabetes.csv`) — 768 patients, 8 features (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`), binary target `Outcome` (0 = non-diabetic, 1 = diabetic). The model itself is intentionally simple here — the notebook's real subject is **model deployment**, not model optimization.

---

## Approach

A trained model living only inside a notebook's memory is useless the moment the kernel restarts — deployment requires **serializing** the fitted pipeline to disk so it can be reloaded later (in an API, a script, a different machine) without retraining. This pair of notebooks covers exactly that: train once, save two ways, then load and predict from a completely fresh session.

Workflow:
1. **Train a simple pipeline** — `StandardScaler` + `LogisticRegression(class_weight="balanced")` — on the diabetes dataset, with a quick sanity-check evaluation (not the focus).
2. **Save the fitted pipeline** two ways: `joblib.dump()` and `pickle.dump()`.
3. **In a separate notebook/session** — reload both saved files (`joblib.load()` and `pickle.load()`), construct a new patient record as a `DataFrame`, and call `.predict()` directly on the loaded pipeline — no re-fitting, no re-scaling by hand.

---

## Layman

Training a model is like baking a cake — expensive, time-consuming, and you don't want to redo it every time someone wants a slice. Saving the model is like freezing that cake: `joblib.dump()` or `pickle.dump()` takes the fully-trained pipeline (scaler included) and freezes it into a single file. Later — in a different kitchen, days later — `load()` thaws it back out, exactly as it was, ready to serve.

The important detail is that the *whole pipeline* gets frozen, not just the model. That matters because a Logistic Regression model expects scaled inputs — if you only saved the model and forgot the scaler, a reloaded model would silently make wrong predictions on raw, unscaled data. Freezing the pipeline as one object means the new patient's numbers get scaled the same exact way the training data was, automatically, every time.

---

## Technical

- **Pipeline saved:** `Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(class_weight="balanced"))])` — the entire fitted pipeline object is serialized, not just the estimator, so the scaler's learned mean/variance travels with it.
- **Quick baseline metrics** (not the focus of this notebook, just a sanity check before saving): 65.1% / 34.9% class split (0 vs 1), 614/154 train/test split (stratified).

| Split | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1 |
|---|---|---|---|---|
| Train | 76.06% | 0.64 | 0.73 | 0.68 |
| Test | 73.38% | 0.60 | 0.70 | 0.65 |

- **`joblib` vs `pickle` — comparison table used in the notebook:**

| Aspect | joblib | pickle |
|---|---|---|
| Speed (large models) | Fast | Slow |
| NumPy array handling | Optimized | Not optimized |
| Memory usage | Efficient | High |
| ML pipelines | Ideal | Works but suboptimal |
| Built-in (no extra install) | No | Yes |
| scikit-learn recommended | Yes | No |

- **Save:**
  - `joblib.dump(model_pipeline, joblib_path)` → `diabetes_model_pipeline.joblib`
  - `pickle.dump(model_pipeline, file)` → `diabetes_model_pipeline_pickle.pkl` (binary mode, `"wb"`)
- **Load (separate notebook/session):**
  - `joblib.load(joblib_path)` and `pickle.load(file)` (binary mode, `"rb"`) both return the identical fitted `Pipeline` object.
  - New patient data is built as a single-row `pd.DataFrame` with the exact same column names/order used in training — this matters because scikit-learn pipelines match by column, and a mismatch would either error or (worse) silently misalign features.
  - `.predict()` is called directly on the loaded pipeline; scaling happens transparently inside it before the model ever sees the raw values.
- **Verified consistency:** both the joblib-loaded and pickle-loaded pipelines predicted the same class (0 — No Diabetes) on the identical test patient (`Pregnancies=2, Glucose=120, BloodPressure=70, SkinThickness=25, Insulin=80, BMI=28.5, DiabetesPedigreeFunction=0.45, Age=35`), confirming the two serialization methods are functionally interchangeable for this pipeline.
- **Data caveat flagged in the notebook (not addressed here):** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` all contain missing values encoded as literal `0` rather than `NaN` — invisible to `.isnull().sum()` (which reports zero nulls), so any production version of this pipeline would need explicit handling (e.g. replace 0 → `NaN` → impute) before those zeros get treated as real physiological values.

---

## Code

```python
# ============================================================
# NOTEBOOK 1 — Train & Save
# ============================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
TARGET_COL = "Outcome"

df = pd.read_csv("diabetes.csv")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# pipeline = scaler + model (saved as ONE object so scaling travels with the model)
model_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced"))
])
model_pipeline.fit(X_train, y_train)

# --- Save with joblib (scikit-learn's recommended choice) ---
from joblib import dump
joblib_path = "model_dir/diabetes_model_pipeline.joblib"
dump(model_pipeline, joblib_path)

# --- Save with pickle (built-in alternative) ---
import pickle
pickle_path = "model_dir/diabetes_model_pipeline_pickle.pkl"
with open(pickle_path, "wb") as file:
    pickle.dump(model_pipeline, file)


# ============================================================
# NOTEBOOK 2 — Load & Predict (fresh session, no retraining)
# ============================================================
import pandas as pd
from joblib import load

loaded_joblib_pipeline = load(joblib_path)

new_data = pd.DataFrame({
    "Pregnancies": [2], "Glucose": [120], "BloodPressure": [70],
    "SkinThickness": [25], "Insulin": [80], "BMI": [28.5],
    "DiabetesPedigreeFunction": [0.45], "Age": [35]
})

# scaling happens automatically inside the pipeline before prediction
prediction = loaded_joblib_pipeline.predict(new_data)
print("✅ No Diabetes" if prediction[0] == 0 else "⚠️ Diabetes")

# --- Equivalent load with pickle ---
import pickle
with open(pickle_path, "rb") as file:
    loaded_pickle_pipeline = pickle.load(file)

prediction = loaded_pickle_pipeline.predict(new_data)
print("✅ No Diabetes" if prediction[0] == 0 else "⚠️ Diabetes")
```

---

## Follow-up

### Approach
The notebook explicitly deferred two things it flagged but didn't fix: the hidden zero-as-missing-value problem, and any real model optimization ("Focusing on Model saving & loading; not on optimization for this video"). Both directly threaten a deployed pipeline's reliability, so they're the natural next layer once the save/load mechanics are solid.

### Layman
A prediction is only as good as the pipeline frozen inside that file. Right now, that frozen pipeline was trained on data where "0 blood pressure" was treated as a real, medically plausible reading rather than what it almost certainly is — a missing measurement. Freezing and reloading a model doesn't fix bad training data; it just preserves whatever the model learned, mistakes included. So before this pipeline is trusted with real patient data, the zeros need to be dealt with *before* the freezing step, not after.

### Technical
Concrete next steps, in rough priority order:
- **Fix the encoded-missing-value problem**: replace `0` with `NaN` in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` (0 is physiologically impossible for all five), add an imputer (median or KNN) inside the pipeline itself — not as a manual preprocessing step outside it — so the fix travels with every future save/load cycle automatically.
- **Version and metadata tracking**: alongside the `.joblib`/`.pkl` file, save a small metadata file (scikit-learn version, training date, feature list/order, CV metrics) — deserialized pickle/joblib files are sensitive to library version mismatches, and a version file makes that failure mode diagnosable instead of silent.
- **Model optimization** (explicitly out of scope in this notebook): the current Logistic Regression is a baseline with no CV, no tuning, and no comparison against tree-based models — the same model-selection + `GridSearchCV` pattern used in the other projects in this series would apply directly here.
- **Input validation on load**: the loaded pipeline currently trusts that `new_data` has the exact right column names/order/dtypes — a production wrapper should validate incoming data (e.g. via a `pydantic` schema or an explicit column-order check) before calling `.predict()`, since a silently misaligned column is a much harder bug to catch than a raised error.