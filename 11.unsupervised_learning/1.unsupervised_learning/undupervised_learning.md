# Section 14 — Unsupervised Learning

> Format for every topic: **Definition → Layman Explanation → Technical Explanation → Examples → Highlights → Brain Map → Code**

---

## 14.1 Unsupervised Learning — Recap

### Definition
Unsupervised Learning deals with **unlabeled data** — the system tries to find hidden patterns or structures in the data **without knowing the correct answers** (no target/label column).

### Layman Explanation
Imagine handing someone a huge box of mixed LEGO pieces with no instruction manual and no picture of the final model. They don't know what "correct" looks like — but they can still start **grouping pieces that look alike** (by color, shape, size) or **notice patterns** (e.g., "these two piece types are almost always used together"). That's unsupervised learning: finding structure with no answer key.

### Technical Explanation
Unlike supervised learning (which learns a mapping from `features → known target`), unsupervised learning only has `features` — no `y`. The algorithm's job is to discover structure on its own.

**Two main types:**

| Type | What it does | Example |
|---|---|---|
| **Clustering** | Groups similar data points based on features, without labels | Grouping customers based on purchasing behavior |
| **Association** | Discovers interesting relationships/rules among variables | "People who buy bread often buy butter" |

**Applications:**
- **Clustering** → Customer Segmentation, Document Clustering, Social Network Analysis, Anomaly Detection, Gene Clustering in Biology
- **Association** → Market Basket Analysis, Recommender Systems, Retail Analytics

**Algorithm families covered in this section:**
- Clustering models: K-Means, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models (GMM)
- Association models: Apriori, Eclat, FP-Growth

### Example
An e-commerce company has millions of transactions with no "customer type" label. Clustering can automatically discover groups like "bulk discount shoppers," "occasional premium buyers," "weekend browsers" — segments no one manually labeled.

### Highlights 🔑
- No target/label column — the biggest structural difference from supervised learning (Section 12–13).
- **Clustering** = grouping similar *rows* (points).
- **Association** = finding relationships between *columns* (items/variables).
- Because there's no "correct answer," evaluation is fundamentally different (often visual, distance-based, or business-driven — not accuracy/F1).

### Brain Map 🧠
```
UNSUPERVISED LEARNING (no labels)
   ├── CLUSTERING     → group similar ROWS       → K-Means, Hierarchical, DBSCAN, GMM
   └── ASSOCIATION    → find relations between COLUMNS/ITEMS → Apriori, Eclat, FP-Growth

Mnemonic: "No answer key → find the groups (clustering) or find the rules (association)."
```

### Code
```python
# Unsupervised learning has no y — only X
from sklearn.cluster import KMeans

# Notice: .fit(X) only, no y!
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)  # no labels needed
labels = model.labels_   # cluster assignments discovered by the algorithm
```

---
