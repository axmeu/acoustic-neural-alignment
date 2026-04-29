from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import PCA
from umap import UMAP
from utils import save_npz, save_csv, is_vowel


FORMANTS = ("F1_mid", "F2_mid")


def lobanov_normalise(df):
    df = df.copy()
    for spk, grp in df.groupby("speaker_id"):
        for formant in FORMANTS:
            if formant not in df.columns:
                continue
            vals = grp[formant].dropna()
            if len(vals) < 2:
                continue
            mu = vals.mean()
            sd = vals.std(ddof=0)
            if sd == 0:
                continue
            df.loc[grp.index, f"{formant}_lob"] = (grp[formant] - mu) / sd
    return df


def apply_pca(vecs, valid_mask, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    coords = np.full((len(vecs), n_components), np.nan, dtype=np.float32)
    coords[valid_mask] = pca.fit_transform(vecs[valid_mask])
    return coords, pca.explained_variance_ratio_, pca.components_, pca.mean_


def apply_umap(vecs, valid_mask, n_neighbors, min_dist):
    coords = np.full((len(vecs), 2), np.nan, dtype=np.float32)
    coords[valid_mask] = UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=42, low_memory=False,
    ).fit_transform(vecs[valid_mask])
    return coords


def normalise_neural(
    npz_path,
    output_dir,
    tag,
    n_pca_clust=50,
    n_pca_lme=5,
    n_umap_neighbors=15,
    umap_min_dist=0.1
):

    output_dir = Path(output_dir)
    data = np.load(npz_path, allow_pickle=True)
    ids = data["ids"]
    vecs = data["vectors"].astype(np.float32)

    valid_mask = ~np.isnan(vecs).any(axis=1)
    n_valid, dim = vecs[valid_mask].shape
    print(f"\n[{tag}] {n_valid}/{len(vecs)} valid vectors  (dim={dim})")

    n_clust = min(n_pca_clust, n_valid, dim)
    coords, evr, comp, mean = apply_pca(vecs, valid_mask, n_clust)
    save_npz(output_dir / f"{tag}_pca{n_clust}.npz",
             ids=ids,
             vectors=coords,
             explained_variance_ratio=evr,
             components=comp,
             mean=mean)

    n_lme = min(n_pca_lme, n_valid, dim)
    coords, evr, comp, mean = apply_pca(vecs, valid_mask, n_lme)
    save_npz(output_dir / f"{tag}_pca{n_lme}_lme.npz",
             ids=ids,
             vectors=coords,
             explained_variance_ratio=evr,
             components=comp,
             mean=mean)

    coords, evr, comp, mean = apply_pca(vecs, valid_mask, 2)
    save_npz(output_dir / f"{tag}_pca2.npz",
             ids=ids,
             vectors=coords,
             explained_variance_ratio=evr,
             components=comp,
             mean=mean)

    coords = apply_umap(vecs, valid_mask, min(n_umap_neighbors, n_valid - 1), umap_min_dist)
    save_npz(output_dir / f"{tag}_umap2.npz", ids=ids, vectors=coords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic",       required=True)
    parser.add_argument("--whisper",        required=True)
    parser.add_argument("--xlsr",           required=True)
    parser.add_argument("--output-dir",     default="outputs")
    parser.add_argument("--n-pca-clust",    type=int, default=50)
    parser.add_argument("--n-pca-lme",      type=int, default=5)
    parser.add_argument("--n-umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist",  type=float, default=0.1)
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    print("\n=== Acoustic normalisation ===")
    df = pd.read_csv(args.acoustic)
    vowel_mask = df["phoneme_label"].str.match(VOWEL_RE, na=False)
    df_vowels = lobanov_normalise(df[vowel_mask].copy())

    df_out = df.copy()
    lob_cols = [f"{f}_lob" for f in FORMANTS if f in df.columns]
    df_out.loc[vowel_mask, lob_cols] = df_vowels[lob_cols].values
    save_csv(output_path / "features_acoustic_norm.csv", df_out)
