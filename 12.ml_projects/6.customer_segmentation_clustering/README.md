# Customer Segmentation Clustering

Customer segmentation project built on the Mall Customers dataset using K-Means clustering. The notebook explores the data, cleans it, scales the features, selects the number of clusters, fits the final model, and profiles the resulting customer segments.

## Dataset

- `Mall_Customers.csv`
- 200 customers
- 5 original columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

## Pipeline And Results

| Stage | What was done | Result |
| --- | --- | --- |
| Data loading | Loaded the CSV and inspected the first rows | Dataset shape: `(200, 5)` |
| Data quality check | Checked nulls, duplicates, and encoded missing values | `0` missing values, `0` duplicate rows |
| Exploratory analysis | Reviewed gender, age, income, and spending distributions | Female customers: `112`, male customers: `88` |
| Feature selection | Kept `Annual Income (k$)` and `Spending Score (1-100)` for clustering | 2 modeling features selected |
| Feature scaling | Standardized the selected features before clustering | Scaled feature matrix created successfully |
| Model evaluation | Trained K-Means and evaluated cluster quality | Silhouette score: `0.555` |
| Final clustering | Fit the final K-Means model and assigned cluster labels | `5` clusters discovered |
| Cluster profiling | Summarized each cluster with size and central tendency | Cluster sizes: `81`, `39`, `22`, `35`, `23` |
| New customer assignment | Tested the segment assignment helper on sample customers | Example assignments returned cluster `0` |

## Cluster Profile Summary

| Cluster | Mean Age | Mean Income | Mean Spending |
| --- | --- | --- | --- |
| 0 | 42.72 | 55.30 | 49.52 |
| 1 | 32.69 | 86.54 | 82.13 |
| 2 | 25.27 | 25.73 | 79.36 |
| 3 | 41.11 | 88.20 | 17.11 |
| 4 | 45.22 | 26.30 | 20.91 |

## How To Use

Open `customer_segmentation_clustering.ipynb` and run the cells top to bottom. The notebook builds the clustering workflow end to end and includes a helper function for assigning a new customer to a segment.

## Notes

- The clustering features are intentionally limited to income and spending score because those variables produce the clearest customer segments.
- The silhouette score and cluster profiles above are taken from the executed notebook outputs.