# Customer Segmentation — Clustering & Silhouette Score

## Definition
Customer segmentation clustering is an unsupervised ML technique that groups customers into distinct segments based on similarity in behavior or attributes (e.g., spend, frequency, recency), without predefined labels. The **silhouette score** is an internal evaluation metric used to judge how well-separated and cohesive those clusters are, most commonly used to pick the optimal number of clusters (`k`) in KMeans.

## Layman
Imagine sorting customers into friend groups at a party based on how they behave — big spenders together, bargain hunters together, one-time visitors together. Clustering does that sorting automatically. The silhouette score is like asking "does each person feel more at home in their assigned group than in any other group?" — a score close to 1 means yes, close to 0 means they're standing awkwardly between two groups, negative means they're probably in the wrong group entirely.

## Technical

**KMeans clustering:**
- Assigns each point to the nearest of `k` centroids, then updates centroids as the mean of assigned points, iterating until convergence.
- Requires `k` to be specified upfront — but the "correct" k is unknown in real segmentation problems.
- Assumes roughly spherical, similarly-sized clusters (Euclidean distance based). Sensitive to feature scaling and outliers.

**Silhouette score:**
For each point `i`:
- `a(i)` = mean intra-cluster distance (cohesion)
- `b(i)` = mean distance to the nearest neighboring cluster (separation)

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- Range: **-1 to 1**
  - **~1** → well-clustered, far from neighboring clusters
  - **~0** → sits on the boundary between two clusters
  - **negative** → likely misassigned
- Averaged across all points to get one score per value of `k`; the `k` that maximizes it is typically chosen.
- Complements the **elbow method** (inertia/WCSS), which can be visually ambiguous — silhouette gives a single comparable number.
- **Limitations:** O(n²) complexity (slow beyond ~50-100k rows); biased toward convex/blob-shaped clusters, which favors KMeans specifically over density-based methods when comparing across algorithms.

## Code

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Always scale before KMeans — it's distance-based
X_scaled = StandardScaler().fit_transform(X)

scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)
```

## Elbow Method

**Why we need it:** Same problem as silhouette — KMeans needs `k` chosen upfront, and the "right" number of segments isn't known in advance. Elbow is the older, more intuitive way to pick `k` by watching how clustering error changes as `k` increases.

**Metric — Inertia / WCSS (Within-Cluster Sum of Squares):**

```
WCSS = Σ (for each cluster) Σ (for each point in cluster) ||point - centroid||²
```

Total squared distance between every point and its own cluster's centroid — how tightly packed the clusters are.

**Mechanism:**
1. Run KMeans across a range of `k` (e.g., 1–10).
2. Record WCSS (`km.inertia_` in sklearn) for each `k`.
3. Plot `k` vs WCSS.
4. Pick `k` at the "elbow" — where the curve stops dropping steeply and flattens.

**Why the curve bends:**
- WCSS always decreases as `k` increases (more centroids → points sit closer to them; at `k = n`, WCSS = 0).
- But the *rate* of decrease drops sharply once `k` passes the true number of natural clusters — beyond that, you're splitting genuinely cohesive groups for little error reduction, hence the flattening "elbow" shape.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('k')
plt.ylabel('WCSS (Inertia)')
```

**Limitation:** The bend is a visual judgment call — on messy real data it's often gradual, not a sharp kink, so different people can read different `k` off the same plot. This is why elbow and silhouette are typically used together: elbow narrows the range, silhouette picks/confirms the exact `k` within it.

## Alternative Models for Customer Segmentation

| Model | When it fits |
|---|---|
| **Hierarchical (Agglomerative)** | Small-to-medium data; dendrogram lets you inspect structure and decide cluster count *after* seeing it |
| **DBSCAN / HDBSCAN** | Non-spherical segments, or outliers/noise (one-off buyers) that shouldn't be forced into a segment; no need to pick k upfront |
| **Gaussian Mixture Models (GMM)** | Soft clustering — probability of belonging to each segment rather than a hard label; useful when a customer straddles two segments |
| **K-Prototypes / K-Modes** | Mixed numeric + categorical features (e.g., spend + membership tier) — plain KMeans doesn't handle categorical well |
| **RFM + KMeans** | Not a new algorithm, but the standard industry framing: engineer Recency/Frequency/Monetary features first, then cluster — much more interpretable for stakeholders |
| **Self-Organizing Maps (SOM)** | Visualizing high-dimensional customer data in 2D before/alongside clustering; less common today |

## Highlights
- Silhouette score answers: *"How do I pick k without labeled data?"*
- KMeans + silhouette is the default combo, but KMeans's spherical-cluster assumption is a real limitation — know when to reach for DBSCAN/GMM instead.
- Always **scale features** before KMeans (distance-based algorithm).
- RFM framing is the most business-interpretable entry point for segmentation and a strong interview answer.

## Brain Map
```
Customer Segmentation
├── Feature Engineering
│   └── RFM (Recency, Frequency, Monetary)
├── Algorithm Choice
│   ├── KMeans (spherical, needs k, fast)
│   ├── Hierarchical (dendrogram, no k needed upfront)
│   ├── DBSCAN/HDBSCAN (density-based, handles noise)
│   ├── GMM (soft/probabilistic assignment)
│   └── K-Prototypes (mixed data types)
└── Evaluation (no ground truth)
    ├── Silhouette Score (cohesion vs separation, -1 to 1)
    └── Elbow Method (WCSS/inertia, visual bend where drop-off flattens)
```

## Follow-up
- Practice explaining *why* KMeans assumes spherical clusters (centroid + Euclidean distance mechanics) — common interview probe.
- Try implementing DBSCAN on the same dataset and compare silhouette scores directly against KMeans to see where the metric favors KMeans unfairly.
- Consider adding a Davies-Bouldin index comparison as a second internal metric, since relying on silhouette alone can be misleading for non-convex clusters.