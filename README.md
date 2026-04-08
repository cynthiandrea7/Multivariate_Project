# Multivariate Project

This repo analyzes S&P 500 stock behavior with:

- Unsupervised clustering (K-means / hierarchical style workflow)
- Volatility forecasting (21-day forward realized volatility target)

## Start Here

Use the staged notebook workflow in:

- [notebooks/README.md](/Users/mintie/Desktop/Multivariate_Project/notebooks/README.md)

Run in order from `01` to `06`.

## Current Scope Rules

- `Weight` is excluded from clustering feature sets for now (to avoid static-weight leakage).
- Time-ordered splits only.
- Each notebook should save outputs to `outputs/` for downstream steps.

## Legacy Notebooks

Older exploratory notebooks remain in project root:

- `feature.ipynb`
- `sp500_clustering_full_mean_std.ipynb`
- `sp500_clustering_full_stability_k_selection.ipynb`

Treat those as reference while migrating to the staged workflow.
