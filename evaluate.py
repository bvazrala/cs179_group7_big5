"""Structure-level evaluation of a learned graph against the known
Big Five item grouping (item trait = first letter of its name)."""

import numpy as np
from networkx import from_numpy_array
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics import adjusted_rand_score

TRAIT_INDEX = {"E": 0, "N": 1, "A": 2, "C": 3, "O": 4}


def within_between_stats(J, items):
    """Fraction of edges / edge weight falling within trait blocks.
    chance_frac_within = expected within fraction for a random pair."""
    A = np.abs(np.asarray(J))
    t = np.array([c[0] for c in items])
    iu = np.triu_indices(len(items), 1)
    w = A[iu]
    same = t[iu[0]] == t[iu[1]]
    nz = w > 0
    total = w.sum()
    return {
        "n_edges": int(nz.sum()),
        "frac_edges_within_trait": float(same[nz].mean()) if nz.any() else 0.0,
        "frac_weight_within_trait": float(w[same].sum() / total) if total > 0 else 0.0,
        "chance_frac_within": float(same.mean()),
    }


def community_ari(J, items):
    """Greedy-modularity communities on |J| compared to trait labels via
    adjusted Rand index."""
    A = np.abs(np.asarray(J))
    G = from_numpy_array(A)
    if G.number_of_edges() == 0:
        return {"ari": 0.0, "n_communities": len(items)}
    comms = greedy_modularity_communities(G, weight="weight")
    pred = np.zeros(len(items), dtype=int)
    for k, c in enumerate(comms):
        for node in c:
            pred[node] = k
    true = np.array([TRAIT_INDEX[c[0]] for c in items])
    return {"ari": float(adjusted_rand_score(true, pred)),
            "n_communities": int(len(comms))}
