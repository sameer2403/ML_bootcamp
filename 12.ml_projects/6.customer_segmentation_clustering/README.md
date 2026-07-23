# Project — Customer Segmentation (Clustering)

**Dataset:** `Mall_Customers.csv` (Kaggle — `vjchoudhary7/customer-segmentation-tutorial-in-python`) — 200 mall customers, 5 columns (`CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`), no target — this is unsupervised.

---

## Approach

Group mall customers into distinct segments based on spending behavior, without any labeled "correct" groups to learn from — the goal is to find structure the business can act on (who to target with premium offers, who's price-sensitive, who's already loyal), not to predict a known outcome.

Workflow:
1. **EDA** — shape, dtypes, missingness (none), duplicates (none), gender distribution, feature distributions, and — critically — a scatter of `Annual Income` vs `Spending Score`, which visually reveals 5 distinct blobs before any model is even run.
2. **Preprocessing** — dropped `CustomerID` (not predictive, saved separately for later joining), selected only `Annual Income (k$)` and `Spending Score (1-100)` as clustering features, standard-scaled both since K-Means is distance-based.
3. **Choosing K — Elbow Method (WCSS):** ran K-Means for K = 1 to 10, plotted inertia, identified the elbow at **K = 5**.
4. **Final clustering:** K-Means with `n_clusters=5`, `n_init=10`.
5. **Evaluation:** Silhouette Score, both at K = 5 and swept across K = 2–10 to sanity-check the elbow choice.
6. **Cluster visualization:** income vs. spending scatter, colored by assigned cluster.
7. **Business profiling:** mean/median `Age`, `Annual Income`, `Spending Score` per cluster, and cluster sizes — turning cluster IDs into business-readable segments.
8. **Inference:** `assign_customer_segment()` to score a brand-new customer against the fitted scaler + model.

---

## Layman

There's no "right answer" being predicted here — nobody labeled these 200 customers as belonging to Group A, B, or C ahead of time. Instead, the algorithm looks at where each customer sits on a chart of "how much they earn" vs. "how much they spend" and notices that people naturally clump into groups: some earn a lot and spend a lot, some earn a lot but barely spend, some earn little but spend freely, and so on.

K-Means, the algorithm used here, works by guessing 5 "center points," assigning every customer to their nearest center, then nudging each center to the middle of its group and repeating until the groups stabilize. The tricky part is choosing *how many* groups to look for — too few and you lump very different customers together; too many and you're just describing noise. The "elbow method" answers this by watching how much tighter the groups get as you add more of them, and stopping right where adding more groups stops helping much — which happened to land on 5 here, matching what the eye could already see in the scatter plot.

---

## Technical

- **Feature selection:** clustering was deliberately restricted to `Annual Income (k$)` and `Spending Score (1-100)` — not `Age` or `Gender` — because the initial income-vs-spending scatter already showed visually separable structure that those two features alone explain; `Age` and `Gender` were reserved for post-hoc cluster profiling instead.
- **Scaling:** `StandardScaler` applied before K-Means, since K-Means is Euclidean-distance-based and both features are on different scales (income in tens of k$, spending on a 1–100 score).
- **Elbow method (WCSS/inertia, K = 1 to 10):**

| K | WCSS |
|---|---|
| 1 | 400.00 |
| 2 | 273.67 |
| 3 | 157.70 |
| 4 | 109.23 |
| **5** | **65.57** |
| 6 | 60.13 |
| 7 | 49.67 |
| 8 | 37.32 |
| 9 | 32.50 |
| 10 | 30.06 |

The steep drop through K = 2–4 flattens sharply after K = 5, marking the elbow.

- **Final model:** `KMeans(n_clusters=5, n_init=10, random_state=42)`.
- **Silhouette Score at K = 5: 0.555** — a respectable score (closer to 1 is better-separated), consistent with visually distinct clusters rather than an artificially forced split. A silhouette sweep across K = 2–10 was also run to cross-check that K = 5 wasn't an arbitrary elbow-method pick.
- **Cluster sizes and business profiles:**

| Cluster | Size | Avg Age | Avg Income (k$) | Avg Spending Score | Segment |
|---|---|---|---|---|---|
| 0 | 81 | 42.7 | 55.3 | 49.5 | Average customers — mid income, mid spending |
| 1 | 39 | 32.7 | 86.5 | 82.1 | **Target segment** — high income, high spending |
| 2 | 22 | 25.3 | 25.7 | 79.4 | Careless spenders — low income, high spending |
| 3 | 35 | 41.1 | 88.2 | 17.1 | Careful high earners — high income, low spending |
| 4 | 23 | 45.2 | 26.3 | 20.9 | Sensible/low-value — low income, low spending |

- **Read on the segments:** Cluster 1 (high income, high spending) is the clearest premium-marketing target; Cluster 3 (high income, low spending) is the most business-interesting group to investigate further — they can afford to spend but currently don't, making them the best candidate for re-engagement campaigns.

---

## Code

```python
# --- Preprocessing ---
customer_id = df["CustomerID"]
df = df.drop(columns=["CustomerID"])

columns_to_select = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[columns_to_select]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Elbow method (choosing K) ---
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(list(k_range), wcss, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method")
plt.show()

# --- Final K-Means clustering ---
K_FINAL = 5

kmeans_final = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)
kmeans_final.fit(X_scaled)
clusters = kmeans_final.predict(X_scaled)

df_clusters = df.copy(deep=True)
df_clusters["clusters"] = clusters
df_clusters["CustomerID"] = customer_id

# --- Evaluation ---
k_means_score = silhouette_score(X_scaled, df_clusters["clusters"])
print("K-Means Clustering - Silhouette Score:", round(k_means_score, 3))

sil_scores = []
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = model.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, cluster_labels))

# --- Business profiling ---
profile_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
cluster_sizes = df_clusters["clusters"].value_counts().sort_index()
cluster_profile_mean = df_clusters.groupby("clusters")[profile_cols].mean()
cluster_profile_median = df_clusters.groupby("clusters")[profile_cols].median()

# --- Inference: assign a new customer to a segment ---
def assign_customer_segment(income_k, spending_score, scaler, model):
    new_point = pd.DataFrame(
        [[income_k, spending_score]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )
    new_point_scaled = scaler.transform(new_point)
    cluster_id = model.predict(new_point_scaled)[0]
    return cluster_id
```

---

## Follow-up

### Approach
This notebook stopped at a clean, well-separated K-Means result on two features — the natural next step isn't tuning K-Means harder, it's testing whether the segmentation holds up under more realistic conditions: more features, an algorithm that doesn't assume round clusters, and a way to validate that the segments are actually stable rather than an artifact of this one 200-row sample.

### Layman
Right now the segmentation only looks at income and spending — it ignores age and gender entirely except to describe the clusters after the fact. But two 25-year-olds and two 55-year-olds with identical income and spending might behave very differently as customers. Bringing age (and possibly gender) into the clustering itself, not just the profiling step afterward, could reveal finer-grained segments — e.g., "young high spenders" vs. "older high spenders" might currently be lumped into the same cluster even though a marketing team would treat them very differently.

### Technical
Concrete next steps, in rough priority order:
- **Include `Age` (and encode `Gender`) as clustering features**, not just profiling features — re-run the elbow method and silhouette analysis on the expanded feature set, since adding dimensions changes both the optimal K and the resulting segment boundaries.
- **Try DBSCAN or Gaussian Mixture Models** as alternatives to K-Means — K-Means assumes roughly spherical, equally-sized clusters, which happens to fit this income/spending scatter well but won't generalize to messier or non-convex customer data.
- **Stability check:** re-run K-Means across multiple random seeds / bootstrap resamples of the 200 customers and check how consistently customers land in the same cluster — a single silhouette score on one fit doesn't confirm the segmentation is robust.
- **Dimensionality reduction for validation:** if more features are added, use PCA or t-SNE to visually confirm the higher-dimensional clusters still separate cleanly, since the current 2D scatter check won't scale past two features.
- **Actionability layer:** turn the 5 profiled segments into named personas with a specific recommended action each (e.g. re-engagement offer for Cluster 3's "high income, low spending" group) so the clustering output plugs directly into a marketing workflow rather than staying descriptive.