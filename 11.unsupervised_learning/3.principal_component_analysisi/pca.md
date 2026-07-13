
## 14.4 Principal Component Analysis (PCA)

### Definition
PCA is a **dimensionality reduction** technique that compresses data with many features into fewer new features (**principal components**), while keeping the **maximum possible information (variance)**.

### Layman Explanation
Imagine photographing a 3D sculpture. A single 2D photo (from the *best* angle) can still capture most of what makes the sculpture recognizable — you lose some depth information, but you keep the essential shape. PCA finds the "best camera angle(s)" for your data — the direction(s) that show the most variation — and re-projects the data onto those fewer axes.

### Technical Explanation

**Why PCA?**
- Too many features (high dimensionality)
- Redundant / correlated information across features
- High computation cost with many dimensions
- Hard to visualize data beyond 2–3 dimensions

**How it works — steps:**
1. **Find Directions** — identify the direction in the feature space along which the data varies the *most* (maximum variance).
2. **Create New Axes** — these directions become the new axes, called **Principal Components (PCs)**. They are:
   - New, transformed features (linear combinations of the originals)
   - Ordered by importance: **PC1** captures the most variance, **PC2** captures the next most (and is orthogonal/perpendicular to PC1), and so on.
3. **Project Data** — the original data points are re-plotted ("projected") onto these new axes, giving a lower-dimensional representation.

Visually: original data in 3D (Variable #1, #2, #3) gets projected onto just PC1 and PC2 — maximizing the variance captured along PC1, and minimizing the leftover residual "spread" along PC2.

**When to Use:**
- High-dimensional data (many columns)
- Correlated features (redundant information)
- Visualization needs (compress to 2D or 3D to plot)
- Speeding up downstream models (fewer features = faster training)

**Limitations:**
- Loses interpretability (a principal component is a *mix* of original features — hard to explain "PC1" in plain business terms)
- Only a **linear** transformation — won't capture non-linear structure well
- Sensitive to feature scaling (features must be standardized first, or high-magnitude features dominate)
- Some information loss is possible (you're compressing, so you never keep 100% of the original variance unless you keep all components)

### Example
A dataset has 50 correlated financial ratios for companies. PCA reduces this to 5 principal components that capture 95% of the total variance — you can now visualize companies on a 2D scatter plot (PC1 vs PC2) and feed just 5 features into a downstream model instead of 50, speeding up training significantly.

### Highlights 🔑
- PCA doesn't *select* existing features — it *creates new ones* (linear combinations) that summarize the old ones.
- **Always scale your features (StandardScaler) before PCA** — otherwise components will be dominated by whichever feature has the largest raw numeric range (ties back to 12.6 Data Leakage principle — fit the scaler on train only).
- PC1 always captures more variance than PC2, PC2 more than PC3, etc. — this is why you can often drop the later components with minimal information loss.
- PCA is for **columns/features** (dimensionality reduction) — contrast with Clustering, which groups **rows**.

### Brain Map 🧠
```
PCA
   1. FIND DIRECTIONS   → axis of maximum variance
   2. CREATE NEW AXES   → Principal Components (PC1 > PC2 > PC3 ... in importance)
   3. PROJECT DATA      → re-plot data onto fewer, information-rich axes

Mnemonic: "PCA = Photograph from the best angle — keep the shape, drop the extra dimensions."
```

### Code
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Always scale first — PCA is sensitive to feature scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)   # fit on train only

# 2. Fit PCA
pca = PCA(n_components=2)   # reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# 3. Check how much variance each component explains
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance captured:", sum(pca.explained_variance_ratio_))

# 4. Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# --- Alternative: keep components until 95% variance is retained ---
pca_auto = PCA(n_components=0.95)   # automatically picks enough PCs for 95% variance
X_pca_auto = pca_auto.fit_transform(X_scaled)
print("Number of components chosen:", pca_auto.n_components_)
```

---

## Section 14 Summary

```
Unsupervised Learning (no labels)
   ├── Clustering    → K-Means / Hierarchical / DBSCAN / GMM   → groups similar ROWS
   ├── Association    → Apriori / Eclat / FP-Growth             → finds rules among ITEMS
   └── PCA            → dimensionality reduction                → compresses COLUMNS/FEATURES
```

- **14.1**: Unsupervised learning has no target/label — it finds structure on its own, via **clustering** (grouping rows) or **association** (relating columns).
- **14.2**: Three clustering intuitions — K-Means (centroid-based, needs K), Hierarchical (tree-based, no K needed upfront), DBSCAN (density-based, auto-detects noise & cluster count).
- **14.4**: PCA compresses many correlated features into fewer, information-dense principal components — always scale before applying, and expect some interpretability trade-off.