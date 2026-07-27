# — Environment Config & Logging for ML Deployment

**Files covered:** `demo_env.py`, `demo_data_loader.py`, `demo_logging.py`, `env_template.txt`, `requirements.txt` (+ `diabetes.csv` as the sample dataset the loader points at).

---

## Approach

Two deployment-hygiene problems that don't show up in a notebook but bite hard in production: **hardcoded paths/secrets scattered across code**, and **print statements that vanish the moment the terminal closes**. This set of scripts fixes both, independently, then combines them in `demo_data_loader.py`.

1. **Environment variables (`python-dotenv`)** — move machine-specific or sensitive values (dataset path, model path, environment name) out of the source code and into a `.env` file that's never committed, loaded at runtime via `load_dotenv()` + `os.getenv()`. `env_template.txt` is the shareable template that documents which variables are expected, without containing real values.
2. **Logging (`logging` module)** — replace `print()` with structured, leveled, timestamped log messages that go to both the terminal and a persistent `app.log` file simultaneously.
3. **`demo_data_loader.py`** combines both: it loads `DATASET_PATH` from the environment instead of hardcoding it, and logs each stage of the load instead of printing silently.
4. **`requirements.txt`** pins exact versions of every dependency these scripts (and the earlier ML pipeline notebooks) rely on, so the environment is reproducible on another machine.

---

## Layman

Imagine handing your notebook to a teammate and the very first cell breaks because it says `CSV_PATH = "/Volumes/MyDrive/..."` — a path that only exists on your laptop. That's the problem environment variables solve: instead of baking your personal file paths (or worse, API keys and passwords) directly into the code, you put them in a separate `.env` file that stays on your machine, and the code just asks "hey, what's `DATASET_PATH` today?" at runtime. Your teammate makes their own `.env` file pointing at their own paths, and the exact same code runs for both of you.

Logging solves a different problem: `print()` statements disappear the second you close the terminal, and they all look the same — you can't tell "just a status update" from "something's actually broken" at a glance. The `logging` module timestamps every message, labels it by severity (INFO, WARNING, ERROR), and — critically — writes it to a file (`app.log`) at the same time it shows it on screen, so you have a permanent record of what happened even after the program exits. That's the difference between debugging blind and debugging with a flight recorder.

---

## Technical

**Environment variables — `demo_env.py`:**
```python
from dotenv import load_dotenv
load_dotenv()  # reads .env in the current working directory, injects into os.environ

dataset_path = os.getenv("DATASET_PATH")
model_path = os.getenv("MODEL_PATH")
environment = os.getenv("ENVIRONMENT")
```
- `load_dotenv()` must run *before* any `os.getenv()` calls — it populates `os.environ` from the `.env` file's key=value lines.
- `env_template.txt` documents the contract (`DATASET_PATH`, `MODEL_PATH`, `ENVIRONMENT`) without real values — the actual `.env` file (not shown here, and correctly not committed to version control) supplies real paths/secrets. `ENVIRONMENT=development` follows the standard pattern of gating behavior by deploy stage (development / staging / production).

**Logging — `demo_logging.py`:**
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),        # prints to terminal
        logging.FileHandler("app.log")  # appends to a persistent file
    ]
)
```
- `level=logging.INFO` means `logging.debug(...)` calls are silently suppressed (DEBUG < INFO in severity) — visible in the demo, where `logging.debug("Debugging code")` produces no output while `info`/`warning`/`error` all do. This is the standard lever for controlling verbosity without touching every log call.

- Dual handlers (`StreamHandler` + `FileHandler`) mean every message goes to both destinations from a single `logging.info(...)` call — no need to print and separately write to a file.

- `%(asctime)s - %(levelname)s - %(message)s` is a minimal but production-reasonable format; a real service would typically add `%(name)s` (module/logger name) for multi-module codebases.

**Combined — `demo_data_loader.py`:**
```python
load_dotenv()
logging.basicConfig(...)  # same config as above

DATASET_PATH = os.getenv("DATASET_PATH")

logging.info("Loading dataset")
df = pd.read_csv(DATASET_PATH)
logging.info("Dataset loaded successfully")
```
- This is the pattern that scales: config comes from the environment, progress/errors go through logging, and the actual business logic (`pd.read_csv`) is the only line that isn't plumbing.
- **Gap worth noting:** there's no error handling around `pd.read_csv(DATASET_PATH)` — if `DATASET_PATH` is `None` (missing `.env`/variable) or the file doesn't exist, this raises an unhandled exception rather than a clean logged error. See Follow-up.

**`requirements.txt`:**
```
numpy==2.4.1
pandas==2.3.3
scikit-learn==1.8.0
joblib==1.5.3
python-dotenv==1.2.1
```
- Exact pinning (`==`, not `>=`) trades flexibility for reproducibility — the standard choice for a project meant to run identically across machines/environments rather than always pull the newest compatible versions.

---

## Code

```python
# ============================================================
# .env  (NOT committed to version control — real values live here)
# ============================================================
# DATASET_PATH="/absolute/path/to/diabetes.csv"
# MODEL_PATH="/absolute/path/to/diabetes_model_pipeline.joblib"
# ENVIRONMENT=development


# ============================================================
# env_template.txt  (committed — documents the contract, no secrets)
# ============================================================
# DATASET_PATH="path/of/dataset"
# MODEL_PATH="path/of/model/file"
# ENVIRONMENT=development


# ============================================================
# demo_env.py  — reading config from the environment
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

dataset_path = os.getenv("DATASET_PATH")
model_path = os.getenv("MODEL_PATH")
environment = os.getenv("ENVIRONMENT")

print("Dataset path:", dataset_path)
print("Model path:", model_path)
print("Environment:", environment)


# ============================================================
# demo_logging.py — structured logging to terminal + file
# ============================================================
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logging.info("Program started")
logging.debug("Debugging code")          # suppressed — level=INFO hides DEBUG
logging.warning("This is a warning message")
logging.error("This is an error message")


# ============================================================
# demo_data_loader.py — env config + logging combined
# ============================================================
import os
import logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

DATASET_PATH = os.getenv("DATASET_PATH")

logging.info("Loading dataset")
df = pd.read_csv(DATASET_PATH)
logging.info("Dataset loaded successfully")

print(df.head())
```

---

## Follow-up

### Approach
The three scripts establish the mechanics correctly but stop short of making them production-safe: none of them handle the failure case where the expected environment variable or file simply isn't there. That's the natural next layer — turning "this works when everything is set up right" into "this fails loudly and usefully when it isn't."

### Layman
Right now, if someone forgets to create their `.env` file, `demo_data_loader.py` doesn't say "hey, you forgot to set `DATASET_PATH`" — it just crashes with a confusing pandas error about a missing filename, because `os.getenv()` silently returned `None` and nobody checked. A more robust version would fail immediately and clearly at startup — "Missing required environment variable: DATASET_PATH" — rather than several lines later with a stack trace that doesn't obviously point back to the real cause.

### Technical
Concrete next steps, in rough priority order:
- **Validate required env vars at startup**: `DATASET_PATH = os.getenv("DATASET_PATH")` followed by `if not DATASET_PATH: raise EnvironmentError("Missing required env var: DATASET_PATH")`, so failures surface immediately with a clear message instead of downstream inside `pd.read_csv`.
- **Wrap the load in try/except with logging**: catch `FileNotFoundError`/`pd.errors.ParserError` around `pd.read_csv`, log via `logging.error(...)` with the exception, and exit or re-raise deliberately rather than letting an unhandled traceback be the only signal.
- **Environment-gated behavior**: `ENVIRONMENT` is read but never used — a realistic next step is branching on it (e.g. more verbose `DEBUG`-level logging in `development`, `WARNING`-level-only in `production`, or pointing at different dataset/model paths per environment).
- **Rotating log files**: `FileHandler("app.log")` will grow unbounded — `logging.handlers.RotatingFileHandler` or `TimedRotatingFileHandler` caps file size/age, which matters the moment this runs continuously rather than as a one-off script.
- **Named loggers instead of the root logger**: `logging.info(...)` at module level uses the root logger; a multi-module project typically uses `logger = logging.getLogger(__name__)` per file so log messages can be filtered/attributed by source module.
- **`.gitignore` the real `.env`**: worth stating explicitly even though it's implied — only `env_template.txt` should ever be committed; the actual `.env` (with real paths/secrets) needs a `.gitignore` entry to avoid accidental commits.