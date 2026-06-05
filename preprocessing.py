"""Preprocessing for the openpsychometrics BIG5 dataset.

Input: BIG5/data.csv (tab-separated). 50 Likert items (1-5) named
E1..E10, N1..N10, A1..A10, C1..C10, O1..O10, plus demographic columns.
Unanswered items are coded 0.

Outputs (in --outdir):
  clean_responses.csv   integer 1-5 item matrix, rows with missing values dropped
  binary_responses.csv  0/1 item matrix
  splits.npz            X_int_train/X_int_test/X_bin_train/X_bin_test/items
  meta.json             item names, trait map, keying, settings, split sizes
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

TRAITS = {"E": "Extraversion", "N": "Neuroticism", "A": "Agreeableness",
          "C": "Conscientiousness", "O": "Openness"}

# Standard IPIP 50-item reverse-keyed items.
REVERSE_KEYED = {
    "E2", "E4", "E6", "E8", "E10",
    "N2", "N4",
    "A1", "A3", "A5", "A7",
    "C2", "C4", "C6", "C8",
    "O2", "O4", "O6",
}

ITEM_RE = re.compile(r"^[ENACO]\d+$")
LIKERT_MIN, LIKERT_MAX = 1, 5


def load_raw(path, sep=None):
    """Load the raw file, sniffing the delimiter; fall back to tab."""
    df = pd.read_csv(path, sep=sep, engine="python")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t", engine="python")
    return df


def get_item_columns(df):
    """Return the 50 item columns in canonical trait order."""
    items = [c for c in df.columns if ITEM_RE.match(str(c))]
    order = {t: i for i, t in enumerate("ENACO")}
    items.sort(key=lambda c: (order[c[0]], int(c[1:])))
    if len(items) != 50:
        print(f"[warn] expected 50 item columns, found {len(items)}: {items}")
    return items


def clean(df, items):
    """Restrict to items, coerce to numeric, drop rows with missing or
    out-of-range (non 1-5) values."""
    X = df[items].apply(pd.to_numeric, errors="coerce")
    X = X.mask((X < LIKERT_MIN) | (X > LIKERT_MAX))  # 0 = unanswered
    before = len(X)
    X = X.dropna(axis=0, how="any").astype(int)
    print(f"[clean] dropped {before - len(X)} rows -> {len(X)} remain")
    return X.reset_index(drop=True)


def reverse_score(X, items):
    """Flip reverse-keyed items on the 1-5 scale (6 - x)."""
    X = X.copy()
    flipped = [c for c in items if c in REVERSE_KEYED]
    for c in flipped:
        X[c] = (LIKERT_MIN + LIKERT_MAX) - X[c]
    print(f"[reverse] reverse-scored {len(flipped)} items")
    return X


def binarize(X, method="gt_neutral", values="01"):
    """Map 1-5 responses to binary.

    method: gt_neutral (x>3), ge_neutral (x>=3), median (x>=column median).
    values: '01' -> {0,1}, 'pm1' -> {-1,+1}.
    """
    if method == "gt_neutral":
        B = (X > 3).astype(int)
    elif method == "ge_neutral":
        B = (X >= 3).astype(int)
    elif method == "median":
        B = (X >= X.median(axis=0)).astype(int)
    else:
        raise ValueError(f"unknown binarize method: {method}")
    if values == "pm1":
        B = B.replace({0: -1, 1: 1})
    elif values != "01":
        raise ValueError("values must be '01' or 'pm1'")
    frac = B.replace({-1: 0}).mean(axis=0)
    print(f"[binarize] method={method} values={values} | "
          f"frac-positive min={frac.min():.2f} max={frac.max():.2f}")
    return B


def split(n, test_size=0.2, seed=0):
    """Return boolean train/test masks over n rows."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * (1 - test_size)))
    train = np.zeros(n, dtype=bool)
    train[idx[:cut]] = True
    return train, ~train


def main():
    p = argparse.ArgumentParser(description="Preprocess the BIG5 dataset.")
    p.add_argument("--input", default="BIG5/data.csv")
    p.add_argument("--outdir", default="processed")
    p.add_argument("--no-reverse", action="store_true")
    p.add_argument("--binarize", default="gt_neutral",
                   choices=["gt_neutral", "ge_neutral", "median"])
    p.add_argument("--binary-values", default="01", choices=["01", "pm1"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_raw(args.input)
    items = get_item_columns(df)
    X = clean(df, items)
    if not args.no_reverse:
        X = reverse_score(X, items)
    B = binarize(X, method=args.binarize, values=args.binary_values)

    train, test = split(len(X), test_size=args.test_size, seed=args.seed)
    print(f"[split] train={train.sum()}  test={test.sum()}")

    X.to_csv(os.path.join(args.outdir, "clean_responses.csv"), index=False)
    B.to_csv(os.path.join(args.outdir, "binary_responses.csv"), index=False)
    np.savez_compressed(
        os.path.join(args.outdir, "splits.npz"),
        X_int_train=X.values[train], X_int_test=X.values[test],
        X_bin_train=B.values[train], X_bin_test=B.values[test],
        items=np.array(items),
    )
    meta = {
        "n_respondents": int(len(X)),
        "items": items,
        "trait_of_item": {c: TRAITS[c[0]] for c in items},
        "reverse_keyed_applied": (not args.no_reverse),
        "reverse_keyed_items": sorted(REVERSE_KEYED),
        "binarize_method": args.binarize,
        "binary_values": args.binary_values,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] wrote outputs to {args.outdir}/")


if __name__ == "__main__":
    main()
