"""Linear probes with the two controls this design cannot do without.

n=128 contexts in d=3584 dimensions is heavily overparameterised: an unregularised
probe separates *any* binary labelling of this data, including random ones. Two
controls are therefore mandatory, and both are enforced here rather than offered:

1. GROUPED cross-validation. Contexts from one domain share almost all of their
   tokens, so a plain 80/20 split lets the probe memorise the domain and score
   near-perfectly on a held-out sibling. Folds are split on `groups` (domain), so
   every context of a domain lands on the same side of the split.
2. A LABEL-PERMUTATION null. The chance level for a probe at this n/d ratio is an
   empirical quantity, not 0.5. Labels are shuffled GLOBALLY (class balance preserved,
   group structure not) and the identical grouped-CV procedure is rerun to get the null
   distribution, against which the observed AUC is reported.

A third control belongs alongside these and is not enforced here because it is a
property of the result rather than of the fit: the LAYER-1 AUC. Layer 1 can only see
token identity, so a probe whose layer-1 value is close to its peak is reading surface
form. Report it next to every peak AUC. And for any direction that will be steered,
report the SPLIT-HALF COSINE (fit on disjoint halves of the groups, compare): at n~91
this ran to only 0.33, which is fine for classification and far too noisy to inject.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def make_probe(C: float = 0.01):
    """Standardise then fit L2 logistic regression. Strong default regularisation."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=4000, solver="lbfgs"),
    )


def cv_auc(X, y, groups, C=0.01, n_splits=8, return_preds=False):
    """Group-wise cross-validated AUC. Returns pooled out-of-fold AUC."""
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        p = make_probe(C).fit(X[tr], y[tr])
        oof[te] = p.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    auc = roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) > 1 else np.nan
    return (auc, oof) if return_preds else auc


def permutation_null(X, y, groups, C=0.01, n_perm=200, seed=0, n_splits=8):
    """Null distribution of grouped CV AUC under label shuffling.

    Labels are permuted across the whole set (preserving class balance), then the
    identical grouped-CV procedure is rerun. This is the honest chance level.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    out = np.empty(n_perm)
    for i in range(n_perm):
        out[i] = cv_auc(X, rng.permutation(y), groups, C=C, n_splits=n_splits)
    return out


def summarise(name, X, y, groups, C=0.01, n_perm=200, seed=0, n_splits=8):
    """Peak AUC against its permutation null.

    `n_splits` is forwarded to both the fit and the null so the two are computed by the
    identical procedure. It previously was not accepted here, pinning every call to 8
    folds regardless of how many groups the data had -- fine at 16 domains, wrong the
    moment the domain count changes.
    """
    auc = cv_auc(X, y, groups, C=C, n_splits=n_splits)
    null = permutation_null(X, y, groups, C=C, n_perm=n_perm, seed=seed, n_splits=n_splits)
    null = null[~np.isnan(null)]
    p = float((null >= auc).mean()) if len(null) else np.nan
    return {
        "probe": name, "auc": float(auc),
        "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
        "p_perm": p, "n": len(y), "d": X.shape[1],
        "n_splits": min(n_splits, len(np.unique(np.asarray(groups)))),
        "beats_null": bool(auc > np.percentile(null, 95)),
    }


def split_half_cosine(A, y, groups, layer, n_iter=30, seed=0, method="logistic"):
    """Reproducibility of a DIRECTION across disjoint halves of the groups.

    Classification tolerates a noisy direction; steering does not, because steering
    injects the vector itself. Gate any steering experiment on this. Measured 0.33 on
    the 16-domain set (n=91), which is mostly noise.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    y = np.asarray(y)
    uniq = np.unique(groups)
    vals = []
    for _ in range(n_iter):
        perm = rng.permutation(uniq)
        h1, h2 = set(perm[:len(perm) // 2]), set(perm[len(perm) // 2:])
        m1 = np.array([g in h1 for g in groups])
        m2 = np.array([g in h2 for g in groups])
        if len(np.unique(y[m1])) < 2 or len(np.unique(y[m2])) < 2:
            continue
        d1 = direction(A[m1][:, layer, :], y[m1], method=method)
        d2 = direction(A[m2][:, layer, :], y[m2], method=method)
        vals.append(abs(cos(d1, d2)))
    return float(np.mean(vals)) if vals else np.nan


def direction(X, y, C=0.01, method="logistic"):
    """Unit-norm direction separating y==1 from y==0, in raw activation space."""
    X, y = np.asarray(X), np.asarray(y)
    if method == "diffmean":
        v = X[y == 1].mean(0) - X[y == 0].mean(0)
    else:
        pipe = make_probe(C).fit(X, y)
        sc, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
        # undo standardisation so the direction lives in raw activation space
        v = (lr.coef_[0] / sc.scale_)
    n = np.linalg.norm(v)
    return v / n if n else v


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else np.nan
