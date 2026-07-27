#  (Production Version) — `training.py` / `prediction.py`

**Files covered:** `training.py`, `prediction.py`, `env_template.txt`, `requirements.txt`, plus the earlier `16_2_1_save_trained_model.ipynb` / `16_2_2_load_trained_model.ipynb` as the exploratory precursor these scripts formalize.

---

## Approach

This is the notebook-to-script graduation step: the same diabetes classification pipeline from the earlier notebooks (`StandardScaler` + `LogisticRegression(class_weight="balanced")`), but rebuilt as two standalone, runnable Python scripts — `training.py` and `prediction.py` — that fix nearly every gap the notebook version left open: hardcoded paths, no error handling, no path-existence checks, log files that overwrite instead of append cleanly.

1. **`env_template.txt`** now defines a full config contract — not just a dataset path, but `PROJECT_ROOT`, `MODEL_DIR`/`MODEL_NAME`, `LOG_DIR`/`LOG_NAME`, and even training hyperparameters (`TARGET_COL`, `TEST_SIZE`, `RANDOM_STATE`) — so nothing about *how* the model trains is hardcoded in the script anymore.
2. **`training.py`** loads config from the environment, builds all paths with `pathlib.Path`, creates output directories if missing, trains the pipeline, logs every stage (to terminal + file), evaluates, and saves the model with `joblib.dump`.
3. **`prediction.py`** mirrors that structure for inference: load config → load the saved model → predict on a hardcoded example patient → log the result.
4. Both scripts wrap their entire body in `try/except`, using `logging.exception()` in the `except` block to capture the full traceback in the log file, not just a one-line error message.
5. **The notebooks' closing comparison** now adds a `Security` row (`joblib`/`pickle` both unsafe for untrusted files) and an explicit course decision: **joblib is the one used going forward** — which is exactly what `training.py`/`prediction.py` use.

---

## Layman

The earlier notebooks proved the *mechanics* work — save a model, load it, predict. These two scripts are what happens when that idea has to survive contact with reality: what if the log folder doesn't exist yet? What if training crashes halfway through — does anyone find out, or does it fail silently? What if someone runs this on a server with a completely different folder structure than your laptop?

`training.py` and `prediction.py` answer all of that. Every path is built from a handful of environment variables, so moving to a new machine means editing one `.env` file, not hunting through code. Every stage logs a timestamped line to both the screen and a file, so if something breaks at 3am on a server nobody's watching, there's a written record of exactly what happened and why (`logging.exception` even captures the full error trace, not just the message). And the whole thing is wrapped so that a failure doesn't die silently — it prints a clear `❌` message, logs the full crash, and still raises the error so the failure is impossible to miss.

---

## Technical

**Config contract — `env_template.txt`:**
```
PROJECT_ROOT="project_root_directory_path"
DATASET_NAME=diabetes.csv
MODEL_DIR=model_dir
LOG_DIR=logs
MODEL_NAME="diabetes_pipeline_trained.joblib"
LOG_NAME="app.log"
TARGET_COL="Outcome"
TEST_SIZE=0.2
RANDOM_STATE=42
```
- All paths are composed relative to a single `PROJECT_ROOT`, using `pathlib.Path` division (`PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")`) rather than string concatenation — cross-platform-safe (no manual `/` vs `\` handling) and far more readable than raw `os.path.join` chains.
- Training hyperparameters (`TEST_SIZE`, `RANDOM_STATE`, `TARGET_COL`) are now environment-driven too, not just file locations — env vars arrive as strings, so `training.py` explicitly casts (`float(os.getenv("TEST_SIZE"))`, `int(os.getenv("RANDOM_STATE"))`) — a detail easy to miss and a common source of bugs if skipped.

**`training.py` — key structural choices:**
- `MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)` and the equivalent for `LOG_PATH` — output directories are created if missing rather than assumed to exist, which is exactly the kind of thing that silently breaks a first-time run on a new machine.
- Logging captures the full lifecycle: dataset shape after load, training completion, train/test accuracy, and the *entire* classification report (multi-line, via f-string) for both splits — not just a final accuracy number, so a later debugging session has the full picture without re-running training.
- `try/except Exception as e` wraps the whole function; on failure, `logging.exception(...)` logs the message **and** the full stack trace (this is `logging.exception`'s specific behavior — it must be called from inside an `except` block), then `raise` re-raises so the script still exits with a non-zero status rather than pretending to succeed.
- **Pipeline is unchanged from the earlier notebook baseline** — same `StandardScaler` + `LogisticRegression(class_weight="balanced")`, same `diabetes.csv`. This script wasn't executed as part of this write-up, but given identical code and data, it should reproduce the earlier notebook's numbers: ~76.06% train / 73.38% test accuracy.

**`prediction.py` — key structural choices:**
- `predict(model, input_data: dict)` is a small, separately-testable function — takes a dict, wraps it in a single-row DataFrame, returns the raw prediction — decoupled from the logging/config plumbing in `main()`. This separation matters if these scripts are ever imported as modules (e.g. by an API layer) rather than run standalone.
- Same config-loading and path-building pattern as `training.py`, reading `MODEL_DIR`/`MODEL_NAME` to reconstruct the exact path `training.py` saved to — the two scripts are implicitly coupled through the shared `.env` contract, which is the intended design (train once, predict many times, same config source).
- Same `try/except` + `logging.exception` + `raise` pattern as training, for the same reason: fail loud, fail logged, fail with a real exit code.

**`requirements.txt`** (unchanged from the notebook-era version): `numpy==2.4.1`, `pandas==2.3.3`, `scikit-learn==1.8.0`, `joblib==1.5.3`, `python-dotenv==1.2.1` — exact pins for reproducibility.

**Updated joblib vs. pickle table (from the refreshed notebook):**

| Aspect | joblib | pickle |
|---|---|---|
| Speed (large models) | Fast | Slow |
| NumPy array handling | Optimized | Not optimized |
| Memory usage | Efficient | High |
| ML pipelines | Ideal | Works but suboptimal |
| Built-in | No | Yes |
| scikit-learn recommended | Yes | No |
| Security | Unsafe for untrusted files | Unsafe for untrusted files |

Both formats are called out as **unsafe for untrusted files** — loading a `.joblib`/`.pkl` file executes arbitrary code paths during deserialization, so a model file should be treated with the same trust level as executable code, never loaded from an unverified source. The course's stated decision: **joblib going forward**, matching what both scripts use.

---

## Code

```python
# ============================================================
# training.py
# ============================================================
import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_model():
    try:
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)]
        )

        logging.info("Training script started")
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape {df.shape}")

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced"))
        ])
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")

        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        logging.info(f"Train Accuracy: {train_acc:.3f}")
        logging.info(f"Test Accuracy: {test_acc:.3f}")
        logging.info(f"Train Classification Report: \n{classification_report(y_train, pipeline.predict(X_train))}")
        logging.info(f"Test Classification Report: \n{classification_report(y_test, pipeline.predict(X_test))}")

        dump(pipeline, MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")
        logging.info("Training script finished")

    except Exception as e:
        print(f"Training failed: {e}")
        logging.exception(f"Training script failed: {e}")
        raise


if __name__ == "__main__":
    train_model()


# ============================================================
# prediction.py
# ============================================================
import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load


def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    return model.predict(df)[0]


def main():
    try:
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)]
        )

        loaded_model = load(MODEL_PATH)
        logging.info("Model loaded")

        input_data = {
            "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70,
            "SkinThickness": 25, "Insulin": 80, "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.45, "Age": 35
        }
        prediction = predict(model=loaded_model, input_data=input_data)

        if prediction == 1:
            print("⚠️ Model predicts: Diabetes")
            logging.info("Prediction result: Diabetes")
        else:
            print("✅ Model predicts: No Diabetes")
            logging.info("Prediction result: No Diabetes")

        logging.info("Prediction script finished")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        logging.exception(f"Prediction script failed: {e}")
        raise


if __name__ == "__main__":
    main()
```

---

## Follow-up

### Approach
These scripts close out almost every gap flagged in the earlier notebook write-up — env validation is now implicit (missing vars surface as clear crashes with full tracebacks, not silent `None`s), error handling exists, and log directories self-create. What's left is less about robustness and more about making these two scripts behave like a real, testable, CLI-friendly pair rather than two files with a hardcoded `main()`.

### Layman
Right now, `prediction.py` always predicts on the exact same hardcoded patient — there's no way to run it on a different patient without editing the source code. The next natural step is letting someone pass in a patient's numbers from the command line or a file, so the script becomes a genuinely reusable tool rather than a fixed demo. Similarly, nothing currently checks that `PROJECT_ROOT` in the `.env` file actually points at a real folder before the script tries to use it — a typo there produces a confusing path error deep inside pandas or joblib instead of a clear "your PROJECT_ROOT is wrong" message right at startup.

### Technical
Concrete next steps, in rough priority order:
- **CLI arguments for `prediction.py`**: replace the hardcoded `input_data` dict with `argparse` (or accept a JSON/CSV file path) so the script can score arbitrary patients without code edits — this is also the natural seam for wrapping it behind a FastAPI/Flask endpoint later.
- **Explicit env-var validation before use**: `PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))` will happily build a `Path(None)` and fail confusingly downstream if the var is unset — a guard clause (`if not os.getenv("PROJECT_ROOT"): raise EnvironmentError(...)`) at the top of both `main()`/`train_model()` would surface config mistakes immediately and clearly.
- **Schema validation on prediction input**: `predict()` trusts `input_data` has exactly the right keys in a form pandas can use — a `pydantic` model (or even a manual key-check) would catch a missing/misspelled field before it silently produces a garbage prediction.
- **Unit tests around `predict()`**: since `predict()` is already cleanly separated from I/O and config, it's the easiest function in either script to unit test directly (e.g. with `pytest` + a small fixture model) — worth doing before this graduates further toward a real deployment.
- **Rotating logs**: both scripts still use a plain `FileHandler`, which will grow `app.log` unbounded across repeated runs — `RotatingFileHandler`/`TimedRotatingFileHandler` becomes worth adding once this runs on a schedule rather than ad hoc.
- **Config validation with a schema library**: as `env_template.txt` grows (now 9 variables), manually casting each one (`float(...)`, `int(...)`) in the script body gets error-prone — a small `pydantic-settings` or `dataclass`-based config loader would centralize parsing/validation and fail with one clear error listing everything wrong, rather than one `KeyError`/`ValueError` at a time.