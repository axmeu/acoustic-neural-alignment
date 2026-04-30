from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


LAYER_FILES = {
    "pca_clust": "pca_clust.npz",
    "pca_lme": "pca_lme.npz",
    "pca2": "pca2.npz",
    "umap2": "umap2.npz",
}


def load_layer(layer_dir):
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
    vec_df["token_id"] = ids
    if "token_id" not in meta.columns:
        raise ValueError("meta must contain token_id for safe merge")
    return meta.merge(vec_df, on="token_id", how="inner")


def mantel_test(D1, D2, n_perm=499):
    idx = np.triu_indices(D1.shape[0], k=1)
    v1 = stats.rankdata(D1[idx])
    v2 = stats.rankdata(D2[idx])
    r_obs, _ = stats.pearsonr(v1, v2)
    rng = np.random.default_rng(42)
    n = D1.shape[0]
    count = sum(
        1 for _ in range(n_perm)
        if stats.pearsonr(
            stats.rankdata(D1[np.ix_(p := rng.permutation(n), p)][idx]),
            v2)[0] >= r_obs
    )
    return float(r_obs), (count + 1) / (n_perm + 1)


def bootstrap_ci(df, stat_fn,
                 n_boot=2000, ci=0.95,
                 speaker_col="speaker_id",
                 seed=42):

    rng = np.random.default_rng(seed)
    speakers = df[speaker_col].unique()
    boot = []
    for _ in range(n_boot):
        samp = rng.choice(speakers, size=len(speakers), replace=True)
        resampled = pd.concat(
            [df[df[speaker_col] == s] for s in samp], ignore_index=True)
        try:
            boot.append(stat_fn(resampled))
        except Exception:
            pass
    a = 1 - ci
    return (float(np.nanpercentile(boot, 100 * a / 2)),
            float(np.nanpercentile(boot, 100 * (1 - a / 2))))


def centroid_cosine_dist(v1, v2):
    c1, c2 = v1.mean(axis=0), v2.mean(axis=0)
    n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(1 - np.dot(c1, c2) / (n1 * n2))


def confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h, angle=theta,
                  linewidth=1.5, fill=False, **kwargs)
    ax.add_patch(ell)
