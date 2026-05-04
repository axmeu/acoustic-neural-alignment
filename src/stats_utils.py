from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import matplotlib.pyplot as plt


LAYER_FILES = {
    "pca_clust": "pca_clust.npz",
    "pca_lme": "pca_lme.npz",
    "pca2": "pca2.npz",
    "umap2": "umap2.npz",
}


def load_layer(layer_dir):
    """Load all PCA/UMAP variants for a given neural layer directory."""
    d = Path(layer_dir)
    out = {"name": d.name}
    for key, fname in LAYER_FILES.items():
        p = d / fname
        if p.exists():
            data = np.load(p, allow_pickle=True)
            out[key] = (data["ids"], data["vectors"].astype(np.float32))
        else:
            out[key] = None
    return out


def save_fig(fig, path, dpi=300, tight=True, close=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None)
    if close:
        plt.close(fig)


def merge(meta, ids, vecs):
    if ids is None or vecs is None:
        return pd.DataFrame()
    n_dim = vecs.shape[1]
    pc_cols = [f"pc_{k}" for k in range(n_dim)]
    vec_df = pd.DataFrame(vecs, columns=pc_cols)
    vec_df["phoneme_id"] = ids
    if "phoneme_id" not in meta.columns:
        raise ValueError("meta must contain phoneme_id for merge")
    return meta.merge(vec_df, on="phoneme_id", how="inner")


def mantel_test(D1, D2, n_perm=5000, seed=42, two_sided=True):
    assert D1.shape == D2.shape, "D1 and D2 must have the same shape"
    n = D1.shape[0]
    idx = np.triu_indices(n, k=1)

    v2 = stats.rankdata(D2[idx])
    r_obs, _ = stats.pearsonr(stats.rankdata(D1[idx]), v2)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        D1_perm = D1[np.ix_(p, p)]
        v1_perm = stats.rankdata(D1_perm[idx])
        r_perm, _ = stats.pearsonr(v1_perm, v2)
        if two_sided:
            if abs(r_perm) >= abs(r_obs):
                count += 1
        else:
            if r_perm >= r_obs:
                count += 1

    p_value = (count + 1) / (n_perm + 1)
    return float(r_obs), float(p_value)


def bootstrap_ci(df, stat_fn,
                 n_boot=2000, ci=0.95,
                 speaker_col="speaker_id",
                 seed=42,
                 verbose=True):

    rng = np.random.default_rng(seed)
    speakers = df[speaker_col].unique()
    groups = {s: g for s, g in df.groupby(speaker_col)}

    boot = []
    n_exceptions = 0
    for _ in range(n_boot):
        samp = rng.choice(speakers, size=len(speakers), replace=True)
        resampled = pd.concat([groups[s] for s in samp], ignore_index=True)
        try:
            boot.append(stat_fn(resampled))
        except Exception:
            n_exceptions += 1

    boot_arr = np.array(boot, dtype=float)
    valid = ~np.isnan(boot_arr)
    n_valid = int(valid.sum())

    if verbose:
        if n_exceptions > 0.05 * n_boot:
            print(f"[WARN] bootstrap_ci: {n_exceptions}/{n_boot} resamples raised exceptions")
        if n_valid < 0.5 * n_boot:
            print(f"[WARN] bootstrap_ci: only {n_valid}/{n_boot} valid values "
                  f"after filtering NaN, CI may be unreliable")

    if n_valid == 0:
        return (float("nan"), float("nan"))

    a = 1 - ci
    return (float(np.nanpercentile(boot_arr, 100 * a / 2)),
            float(np.nanpercentile(boot_arr, 100 * (1 - a / 2))))


def centroid_cosine_dist(v1, v2):
    c1, c2 = v1.mean(axis=0), v2.mean(axis=0)
    n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(1 - np.dot(c1, c2) / (n1 * n2))


def confidence_ellipse(x, y, ax, confidence=0.95, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    scale = np.sqrt(chi2.ppf(confidence, df=2))
    w, h = 2 * scale * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h, angle=theta,
                  linewidth=1.5, fill=False, **kwargs)
    ax.add_patch(ell)