# 16.3 — Streamlit Deployment: `app.py` / `predictor.py`

**Files covered:** `app.py`, `predictor.py`, `env_template.txt`, `requirements.txt` — the final layer on top of `training.py`/`prediction.py`: a browser-based UI a non-technical user can actually click through.

---

## Approach

Everything up to this point (`training.py`, `prediction.py`) required someone comfortable running Python scripts. This step wraps the same trained pipeline in a **Streamlit** web app — a form with number inputs and a "Predict" button — so a doctor, recruiter-facing demo, or anyone without a terminal can use the model.

1. **`predictor.py`** — refactors the earlier `prediction.py` into an *importable module*: loads the environment config, sets up logging, and loads the trained model **once at import time** (not per-call), exposing a single clean `predict(input_data: dict)` function for any caller to use — a script, a notebook, or, here, a web app.
2. **`app.py`** — the Streamlit UI: a page config, a two-column form for the 8 patient features (with sane min/max/default bounds per field), a "Predict" button, and a result rendered as a colored success/error banner.
3. **`env_template.txt`** simplifies back down to just the deployment-relevant variables (`PROJECT_ROOT`, `MODEL_DIR`/`MODEL_NAME`, `LOG_DIR`/`LOG_NAME`) — training-only config (`TARGET_COL`, `TEST_SIZE`, `RANDOM_STATE`) is dropped, since this app only ever loads a model, never trains one.
4. **`requirements.txt`** adds `streamlit==1.52.2` on top of the existing pinned stack.

---

## Layman

`training.py` and `prediction.py` proved the model works, but they only speak Python — useless to anyone who can't run a script from a terminal. This step puts a form in front of it: a doctor (or recruiter watching a demo) fills in eight numbers, clicks **Predict**, and gets a plain-language answer — no code, no terminal, no file paths to configure.

Under the hood, `predictor.py` is doing something subtly important: it loads the trained model **once**, when the app starts, not every single time someone clicks Predict. That matters because Streamlit re-runs the whole script top-to-bottom on every interaction — if the model load lived inside the `predict()` function instead of at module level, every click would re-load the model from disk, which is slow and wasteful. Loading it once at import time and reusing it across every prediction is the difference between an app that feels instant and one that lags on every click.

---

## Technical

**`predictor.py` — module-level load pattern:**
```python
load_dotenv()
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
...
model = load(MODEL_PATH)   # runs once, at import time

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    return model.predict(df)[0]
```
- Config loading, logging setup, and the `load(MODEL_PATH)` call all sit at **module level**, outside any function — Python only executes module-level code once per process, on first `import predictor`. Streamlit's rerun-on-every-interaction model means `app.py`'s script body re-executes on each click, but `import predictor` after the first run just returns the already-loaded module from `sys.modules` — the model is not reloaded.
- `predict()` itself is now minimal — just DataFrame construction, `.predict()`, and a log line — because everything else (config, model loading, logging setup) was hoisted to import time.
- The docstring-style commented-out example usage block at the bottom (`# input_data = {...}`) is a leftover of manual testing — harmless, but a candidate for cleanup or moving into an actual test file.

**`app.py` — Streamlit structure:**
- `st.set_page_config(page_title=..., page_icon="🩺", layout="centered")` — must be the first Streamlit call in the script (a Streamlit requirement, not shown as an issue here but worth knowing generally).
- `st.columns(2)` splits the 8 inputs into two visually balanced halves rather than one long vertical form.
- Each `st.number_input(label, min, max, default)` enforces bounds directly in the widget — e.g. `Age` is bounded 1–100, `BMI` 0.0–70.0 — which is lightweight, UI-level input validation the earlier scripts never had (their hardcoded example patient couldn't be out of range by definition; a live form can be).
- `if st.button("🔍 Predict")`: the whole prediction block only runs on click, not on every widget interaction — Streamlit reruns the script on every input change, so gating the actual `predict()` call behind the button prevents wasted inference calls while the user is still adjusting sliders/fields.
- Result rendering: `st.error(...)` for a positive prediction (red banner) and `st.success(...)` for negative (green banner) — a sensible visual mapping (red = concerning result), though worth noting `st.error` is being used for *information display*, not for reporting an application error — a minor semantic overload of the two Streamlit calls that's harmless here but worth being deliberate about in a larger app.
- **Typos present in the current copy** (harmless to functionality, worth a pass before sharing as a portfolio piece): `"The patinet doen't have Diabetes"` → "The patient doesn't have Diabetes".

**`env_template.txt`** — deployment-only slice of the earlier training config:
```
PROJECT_ROOT=/absolute/path/to/your_project
MODEL_DIR=model_dir
MODEL_NAME= diabetes_pipeline_trained.joblib
LOG_DIR=logs
LOG_NAME=app.log
```
(Note: `MODEL_NAME= diabetes_pipeline_trained.joblib` has a stray space after `=` — most `.env` parsers, including `python-dotenv`, will include that leading space in the value unless stripped, which could silently break the path join. Worth trimming.)

**`requirements.txt`** adds one line to the existing pinned stack: `streamlit==1.52.2`.

```

```bash
# run locally
streamlit run app.py
```

---

## Follow-up

### Approach
The app works, but it's a single-file, single-model, unauthenticated form with no input safety net beyond Streamlit's numeric bounds. The next layer is making it robust to bad input, informative about *why* a prediction landed where it did, and closer to something deployable outside a local demo.

### Layman
Right now, if the model loading fails on startup (wrong path, missing file), the whole app just crashes with a raw Python error on screen — not the kind of thing you want a non-technical user staring at. Similarly, the app tells someone "you have diabetes" or "you don't" with total confidence, but a logistic regression actually produces a *probability*, not a yes/no — showing that probability (e.g. "72% likelihood") alongside the verdict gives a much more honest, useful picture than a flat binary label.

### Technical
Concrete next steps, in rough priority order:
- **Show prediction probability, not just the class**: `model.predict_proba(df)[0][1]` gives the actual probability of the positive class — displaying it (e.g. via `st.metric` or a progress bar) is more informative than a binary label and is a one-line change since the pipeline already supports `.predict_proba()`.
- **Graceful failure on model load**: wrap the module-level `load(MODEL_PATH)` in `predictor.py` in a try/except that logs and surfaces a clear Streamlit error (`st.error("Model could not be loaded — contact the administrator")`) instead of letting an unhandled exception crash the whole app on startup.
- **Fix the `.env` value with the stray leading space** (`MODEL_NAME= diabetes...`) before it causes a real path-mismatch bug — worth adding `.strip()` around `os.getenv()` calls defensively regardless.
- **Input sanity warnings, not just hard bounds**: `st.number_input` bounds prevent literally impossible values, but a clinically implausible-but-technically-valid combination (e.g. `Glucose=0`, which the earlier notebooks flagged as an encoded-missing-value sentinel, not a real reading) currently passes through silently — a soft warning banner for suspicious inputs would catch this class of user error.
- **Caching the model load with `st.cache_resource`**: `predictor.py`'s module-level load already avoids reloading on every click, but Streamlit's own `@st.cache_resource` decorator is the idiomatic way to do this within the Streamlit lifecycle (handles hot-reloading during development more gracefully) — worth adopting if this app grows past a single-file demo.
- **Deployment target**: for anything beyond local `streamlit run`, the next real step is packaging this for Streamlit Community Cloud, a Docker container, or a cloud VM, at which point `PROJECT_ROOT`-based absolute paths in `.env` become the main thing to revisit (they assume a specific local filesystem layout).