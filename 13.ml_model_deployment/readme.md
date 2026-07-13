# 📚 ML Notes — Section 16: ML Model Deployment

---

## 16.1 Deployment Overview — Section Roadmap

> **Why Deployment Matters:** A model that lives only in a Jupyter notebook has zero real-world value. Deployment is the bridge between a trained model and a product that users or systems can actually call. This section covers the full journey — from saving a model file to hosting it on a cloud server behind a REST API.

---

### 🗺️ Section Map

```
TRAIN                PACKAGE              SERVE                SHIP
─────────────────────────────────────────────────────────────────
Notebook      →   Save/Load    →   Python Script   →   FastAPI
(experiment)      (.pkl/.joblib)   (train.py          (REST API
                                    predict.py)         endpoint)
                                        │
                                    Streamlit           AWS EC2
                                    (frontend UI)   (cloud host)
                                        │
                                    Git/GitHub
                                   (version control)
```

---

### 📋 Topics in This Section

| # | Topic | What You'll Learn |
|---|---|---|
| 16.1 | **ML Deployment Overview** | Section roadmap (this file) |
| 16.2 | **Saving & Loading Trained ML Models** | Persist models with `joblib` / `pickle` |
| 16.3 | **From Notebooks to Python Scripts** | Convert `.ipynb` → `.py` cleanly |
| 16.4 | **Training and Prediction Scripts** | Modular `train.py` + `predict.py` |
| 16.5 | **Streamlit App for ML Model** | Build a browser UI for predictions |
| 16.6 | **API Development with FastAPI** | REST API fundamentals |
| 16.7 | **Deploy ML Model as API** | Serve predictions over HTTP |
| 16.8 | **FastAPI + Streamlit Integration** | Connect UI frontend to API backend |
| 16.9 | **Git & GitHub Basics** | Version control for ML projects |
| 16.10 | **AWS EC2 — ML API Deployment** | Host ML APIs on a cloud VM |

---

### ⭐ Key Highlights
> 🔑 Deployment = making your model **callable by the world**, not just runnable in your notebook
> 🔑 The deployment stack in this section: `sklearn model → joblib → FastAPI → Streamlit → AWS EC2`
> 🔑 Git/GitHub is the glue that tracks every version of your code across the pipeline

---

*📎 Upload the next PDF (16.2 onwards) and notes will be added below in the same format.*

---
---

<!-- ═══════════════════════════════════════════════════════════
     CONTENT SECTIONS WILL BE APPENDED HERE AS PDFs ARE UPLOADED
     ═══════════════════════════════════════════════════════════ -->