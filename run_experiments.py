"""Experiment runner. Reads processed/splits.npz, fits models, writes
figures/*.png and results.json.

Experiments:
  main            Ising at default C vs independent baseline; structure stats
  learning_curve  held-out fit vs training-set size
  regularization  sparsity / held-out fit vs L1 strength
  factor_k        factor-model test log-likelihood vs number of factors
  loadings        factor loadings heatmap at the best k
  binarization    gt_neutral vs train-median binarization
  comparison      masked-item accuracy: baseline vs Ising vs factor
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from model_ising import IsingModel, IndependentBaseline
from model_factor import FactorModel
from evaluate import within_between_stats, community_ari

TRAIT_NAMES = {"E": "Extraversion", "N": "Neuroticism", "A": "Agreeableness",
               "C": "Conscientiousness", "O": "Openness"}
TRAIT_COLORS = {"E": "tab:red", "N": "tab:purple", "A": "tab:green",
                "C": "tab:blue", "O": "tab:orange"}


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def trait_blocks(items):
    """Contiguous (start, end, trait_letter) runs in item order."""
    blocks, start = [], 0
    for k in range(1, len(items) + 1):
        if k == len(items) or items[k][0] != items[start][0]:
            blocks.append((start, k, items[start][0]))
            start = k
    return blocks


# --------------------------- figures ---------------------------

def fig_adjacency(J, items, path):
    A = np.abs(J)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(A, cmap="viridis", interpolation="nearest")
    blocks = trait_blocks(items)
    for _, e, _ in blocks[:-1]:
        ax.axhline(e - 0.5, color="white", lw=0.8)
        ax.axvline(e - 0.5, color="white", lw=0.8)
    centers = [(s + e - 1) / 2 for s, e, _ in blocks]
    labels = [TRAIT_NAMES[t] for _, _, t in blocks]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Learned pairwise weights |J|, items grouped by trait")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_graph(J, items, path, seed=0, max_edges=150):
    A = np.abs(np.asarray(J))
    iu = np.triu_indices_from(A, 1)
    w = A[iu]
    order = np.argsort(w)[::-1][:min(max_edges, int((w > 0).sum()))]
    G = nx.Graph()
    G.add_nodes_from(range(len(items)))
    for k in order:
        if w[k] <= 0:
            break
        G.add_edge(int(iu[0][k]), int(iu[1][k]), weight=float(w[k]))
    pos = nx.spring_layout(G, seed=seed, weight="weight")
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [TRAIT_COLORS[c[0]] for c in items]
    widths = [2.5 * G[u][v]["weight"] / max(w.max(), 1e-9) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=180)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: items[i] for i in G.nodes()}, font_size=6)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=TRAIT_NAMES[t],
                          markerfacecolor=c, markersize=8)
               for t, c in TRAIT_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_title(f"Learned dependency graph (top {len(G.edges())} edges by |J|)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_learning_curve(res, baseline_acc, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(res["sizes"], res["pseudo_ll"], "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training respondents")
    axes[0].set_ylabel("test pseudo-log-likelihood")
    axes[1].plot(res["sizes"], res["acc"], "o-", label="Ising")
    axes[1].axhline(baseline_acc, color="gray", ls="--", label="independent baseline")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training respondents")
    axes[1].set_ylabel("masked-item accuracy")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_regularization(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(res["C"], res["n_edges"], "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C (inverse L1 strength)")
    axes[0].set_ylabel("number of edges")
    ax = axes[1]
    ax.plot(res["C"], res["pseudo_ll"], "o-", color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse L1 strength)")
    ax.set_ylabel("test pseudo-log-likelihood", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(res["C"], res["acc"], "s--", color="tab:orange")
    ax2.set_ylabel("masked-item accuracy", color="tab:orange")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_factor_k(res, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(res["k"], res["test_ll"], "o-")
    ax.axvline(res["best_k"], color="gray", ls="--")
    ax.annotate(f"best k = {res['best_k']}",
                xy=(res["best_k"], max(res["test_ll"])),
                xytext=(5, -10), textcoords="offset points", fontsize=9)
    ax.set_xlabel("number of factors")
    ax.set_ylabel("test log-likelihood")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_loadings(L, items, path):
    V = L.values if hasattr(L, "values") else np.asarray(L)
    vmax = np.abs(V).max()
    fig, ax = plt.subplots(figsize=(4.5, 8))
    im = ax.imshow(V, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    for _, e, _ in trait_blocks(items)[:-1]:
        ax.axhline(e - 0.5, color="black", lw=0.6)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items, fontsize=5)
    ax.set_xticks(range(V.shape[1]))
    ax.set_xticklabels([f"F{k + 1}" for k in range(V.shape[1])], fontsize=8)
    ax.set_title("Factor loadings")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_comparison(res, path):
    names, vals = list(res.keys()), list(res.values())
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(names, vals, color=["gray", "tab:blue", "tab:green"])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("masked-item accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------- experiments -------------------------

def exp_main(Btr, Bte, items, C, figdir):
    m = IsingModel(C=C).fit(Btr)
    base = IndependentBaseline().fit(Btr)
    out = {
        "C": C,
        "n_edges": m.n_edges,
        "test_pseudo_ll": m.pseudo_log_likelihood(Bte),
        "masked_acc": m.masked_accuracy(Bte),
        "baseline_pseudo_ll": base.pseudo_log_likelihood(Bte),
        "baseline_masked_acc": base.masked_accuracy(Bte),
        "structure": within_between_stats(m.J, items),
        "community": community_ari(m.J, items),
    }
    fig_adjacency(m.J, items, os.path.join(figdir, "adjacency.png"))
    fig_graph(m.J, items, os.path.join(figdir, "graph.png"))
    return out, m, base


def exp_learning_curve(Btr, Bte, C, seed, quick):
    n = len(Btr)
    grid = [400, 1000] if quick else [250, 500, 1000, 2500, 5000, 10000]
    sizes = [s for s in grid if s < n] + [n]
    rng = np.random.default_rng(seed)
    res = {"sizes": sizes, "pseudo_ll": [], "acc": []}
    for s in sizes:
        idx = rng.choice(n, s, replace=False)
        m = IsingModel(C=C).fit(Btr[idx])
        res["pseudo_ll"].append(m.pseudo_log_likelihood(Bte))
        res["acc"].append(m.masked_accuracy(Bte))
    return res


def exp_regularization(Btr, Bte, items, quick):
    Cs = [0.01, 0.1, 1.0] if quick else list(np.logspace(-3, 0, 7))
    res = {"C": Cs, "n_edges": [], "pseudo_ll": [], "acc": [],
           "frac_weight_within_trait": []}
    for C in Cs:
        m = IsingModel(C=C).fit(Btr)
        res["n_edges"].append(m.n_edges)
        res["pseudo_ll"].append(m.pseudo_log_likelihood(Bte))
        res["acc"].append(m.masked_accuracy(Bte))
        res["frac_weight_within_trait"].append(
            within_between_stats(m.J, items)["frac_weight_within_trait"])
    return res


def exp_factor_k(Xtr, Xte, quick):
    ks = [2, 5, 8] if quick else list(range(1, 11))
    ll = [FactorModel(n_factors=k).fit(Xtr).test_log_likelihood(Xte) for k in ks]
    return {"k": ks, "test_ll": ll, "best_k": ks[int(np.argmax(ll))]}


def exp_binarization(Xint_tr, Xint_te, Btr, Bte, items, C):
    med = np.median(Xint_tr, axis=0)
    Mtr = (Xint_tr >= med).astype(int)
    Mte = (Xint_te >= med).astype(int)
    out = {}
    for name, (tr, te) in {"gt_neutral": (Btr, Bte),
                           "train_median": (Mtr, Mte)}.items():
        m = IsingModel(C=C).fit(tr)
        out[name] = {
            "n_edges": m.n_edges,
            "test_pseudo_ll": m.pseudo_log_likelihood(te),
            "masked_acc": m.masked_accuracy(te),
            "frac_weight_within_trait":
                within_between_stats(m.J, items)["frac_weight_within_trait"],
            "community_ari": community_ari(m.J, items)["ari"],
        }
    return out


def exp_comparison(Btr, Bte, ising, base, k):
    fm = FactorModel(n_factors=k).fit(Btr)
    return {"independent": base.masked_accuracy(Bte),
            "ising": ising.masked_accuracy(Bte),
            f"factor_k{k}": fm.masked_accuracy(Bte)}


def main():
    ap = argparse.ArgumentParser(description="Run all experiments.")
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="small grids for a fast end-to-end check")
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    npz = np.load(os.path.join(args.processed, "splits.npz"))
    items = [str(s) for s in npz["items"]]
    Btr, Bte = npz["X_bin_train"], npz["X_bin_test"]
    Xtr, Xte = npz["X_int_train"], npz["X_int_test"]

    results = {}
    print("[1/7] main structure")
    results["main"], ising, base = exp_main(Btr, Bte, items, args.C, args.figdir)
    print("[2/7] learning curve")
    results["learning_curve"] = exp_learning_curve(Btr, Bte, args.C,
                                                   args.seed, args.quick)
    fig_learning_curve(results["learning_curve"],
                       results["main"]["baseline_masked_acc"],
                       os.path.join(args.figdir, "learning_curve.png"))
    print("[3/7] regularization path")
    results["regularization"] = exp_regularization(Btr, Bte, items, args.quick)
    fig_regularization(results["regularization"],
                       os.path.join(args.figdir, "regularization.png"))
    print("[4/7] factor k sweep")
    results["factor_k"] = exp_factor_k(Xtr, Xte, args.quick)
    fig_factor_k(results["factor_k"], os.path.join(args.figdir, "factor_k.png"))
    print("[5/7] factor loadings")
    k = results["factor_k"]["best_k"]
    fm = FactorModel(n_factors=k).fit(Xtr)
    fig_loadings(fm.loadings(items), items,
                 os.path.join(args.figdir, "loadings.png"))
    print("[6/7] binarization comparison")
    results["binarization"] = exp_binarization(Xtr, Xte, Btr, Bte, items, args.C)
    print("[7/7] model comparison")
    results["comparison"] = exp_comparison(Btr, Bte, ising, base, k)
    fig_comparison(results["comparison"],
                   os.path.join(args.figdir, "comparison.png"))

    with open(args.results, "w") as f:
        json.dump(jsonable(results), f, indent=2)
    print(f"[done] figures -> {args.figdir}/  metrics -> {args.results}")


if __name__ == "__main__":
    main()
