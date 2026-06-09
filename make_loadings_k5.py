"""Regenerate the factor-loadings heatmap at k=5 for the report.

The main experiment runner saves loadings at the likelihood-selected k
(k=10 on this data); the report discusses the five-factor solution, so this
produces figures/loadings_k5.png.

Usage:
  python3 make_loadings_k5.py
"""

import numpy as np

from model_factor import FactorModel
from run_experiments import fig_loadings

npz = np.load("processed/splits.npz")
items = [str(s) for s in npz["items"]]

fm = FactorModel(n_factors=5, rotation="varimax").fit(npz["X_int_train"])
fig_loadings(fm.loadings(items), items, "figures/loadings_k5.png")
print("[done] wrote figures/loadings_k5.png")
