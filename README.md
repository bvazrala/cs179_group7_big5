# CS 179 - Algorithms for Probabilistic and Deterministic Graphical Models - Group Project

Group Members: Bala Kausik Vazrala, Aidan Michael Huerta, Jenson Phan

Report link: https://docs.google.com/document/d/1LonjPNZJF08ROklBeoy9eLjJQMX5YC_Ga6r3KMqHl3s/edit?usp=sharing

# Rediscovering the Big Five with Probabilistic Graphical Models

A CS 179 group project (UC Irvine). We learned the conditional dependence
structure among the 50 items of the IPIP Big Five personality inventory using
a pairwise binary Markov random field (an Ising model), and we compare it
against a Gaussian factor analysis baseline. The model is never told that
personality traits exist. The question is whether the familiar organization
into five traits (Extraversion, Neuroticism, Agreeableness, Conscientiousness, 
and Openness) emerges from raw survey responses on its
own.

## What this project shows

Community detection on the learned graph recovers the five traits exactly:
five communities, all 50 items grouped with their designed trait, adjusted
Rand index of 1.00. The learned dependencies are also predictive, raising
masked item accuracy on test respondents from 64.5 percent (an independent
baseline) to 77.6 percent. The factor analysis comparison independently
points to five dominant factors: the test likelihood keeps rising through
ten factors, but its marginal gains collapse after the fifth, and after
varimax rotation each trait block in the loadings is dominated by a single
factor.

Beyond the specific result, the repository is useful as a compact and fully
reproducible case study of structure learning on real data with a known
ground truth. It demonstrates pseudolikelihood neighborhood selection, the
tradeoff between sparsity and interpretability along a regularization path,
learning curves over training set size, robustness checks for modeling
choices, and proper evaluation of a graphical model on a test set. The same
pipeline can serve as a template for any binary response data, such as
surveys or ratings.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

On Windows, activate with `.venv\Scripts\Activate.ps1` instead.

## Data

    mkdir -p data
    curl -L -o data/BIG5.zip https://openpsychometrics.org/_rawdata/BIG5.zip
    unzip data/BIG5.zip -d data/

This produces `data/BIG5/data.csv` along with a codebook that documents the
items. The file uses tab separation; preprocessing detects this
automatically.

## Running the pipeline

    python preprocessing.py --input data/BIG5/data.csv --outdir processed
    python run_experiments.py --processed processed --figdir figures
    python make_loadings_k5.py

The full run takes a few minutes and writes every figure to `figures/` and
every metric to `results.json`. For a quick check that everything works,
add `--quick` to the `run_experiments.py` command.

## Repository contents

    preprocessing.py      cleans the raw file, binarizes responses, writes the train/test split
    model_ising.py        Ising model learned by logistic regressions with L1 regularization
    model_factor.py       Gaussian factor analysis baseline
    evaluate.py           trait block statistics and community recovery
    run_experiments.py    runs all experiments and writes figures and results.json
    make_loadings_k5.py   regenerates the loadings figure at k = 5 with varimax rotation
    report/               project report
    figures/              generated plots
    processed/            preprocessing outputs (regenerated, not committed)
    data/                 raw download (not committed)

## Method notes

Structure learning is pseudolikelihood neighborhood selection: one logistic
regression with L1 regularization for each item, predicting that item from
the other 49, with coefficients symmetrized into an edge weight matrix
(Ravikumar, Wainwright, and Lafferty, 2010). Evaluation uses the conditional
log likelihood and masked item accuracy on test respondents, together with
structural measures of how well the graph matches the trait blocks. The
factor baseline is evaluated by test log likelihood across factor counts
from 1 to 10; loadings are displayed after varimax rotation, which changes
only their presentation and not the fitted model.
