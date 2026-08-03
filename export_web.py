"""Export the learned graph for the portfolio site."""
import json
import numpy as np
from networkx import from_numpy_array
from networkx.algorithms.community import greedy_modularity_communities

from model_ising import IsingModel

C = 1.0          # match whatever run_experiments.py used for the main result
KEEP_EDGES = 220 # target edge count for the render

npz = np.load("processed/splits.npz", allow_pickle=True)
items = [str(s) for s in npz["items"]]
Btr = npz["Btr"]

J = np.asarray(IsingModel(C=C).fit(Btr).J)
n = len(items)
assert J.shape == (n, n), f"expected {n}x{n}, got {J.shape}"

# Same community detection the ARI score was computed from.
comms = greedy_modularity_communities(from_numpy_array(np.abs(J)), weight="weight")
community = np.zeros(n, dtype=int)
for k, c in enumerate(comms):
    for node in c:
        community[node] = k
print(f"{len(comms)} communities found")

# Keep the strongest edges by magnitude, but carry the sign through.
iu = np.triu_indices(n, 1)
mags = np.abs(J[iu])
nonzero = mags[mags > 0]
cut = np.sort(nonzero)[-KEEP_EDGES] if len(nonzero) > KEEP_EDGES else nonzero.min()

edges = []
for i, j in zip(*iu):
    w = J[i, j]
    if abs(w) >= cut and w != 0:
        edges.append({"a": int(i), "b": int(j), "w": float(w)})

peak = max(abs(e["w"]) for e in edges)
for e in edges:
    e["w"] = round(e["w"] / peak, 4)

json.dump(
    {
        "nodes": [
            {"id": i, "label": items[i], "community": int(community[i])}
            for i in range(n)
        ],
        "edges": edges,
    },
    open("big-five-graph.json", "w"),
    indent=1,
)

neg = sum(1 for e in edges if e["w"] < 0)
print(f"{n} nodes, {len(edges)} edges ({neg} negative) -> big-five-graph.json")
