import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm import tqdm
from utils import is_vowel, save_csv, front_back, height
from stats_utils import merge, save_fig
from analyse.descriptive import (F1_LOB, F2_LOB, filter_canonical_vowels)


SCG_COL = "SCG"
DURATION_COL = "duration_ms"

TARGET_CONSONANTS = ["p", "t", "k", "s", "f", "m", "n", "l", "ʁ"]

K_RANGE = range(2, 7)


def _ward_linkage(X, metric):
    if metric == "euclidean":
        return linkage(X, method="ward"), X
    elif metric == "cosine":
        X = X.astype(np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        Xn = X / nrm
        return linkage(Xn, method="ward"), Xn
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def _silhouette(X, labels, metric):
    if len(set(labels)) < 2:
        return np.nan
    sk_metric = "cosine" if metric == "cosine" else "euclidean"
    try:
        return silhouette_score(X, labels, metric=sk_metric)
    except Exception:
        return np.nan


def _evaluate_vowel_clustering(centroids_df, metric, tag, out, ari_rows):
    if centroids_df.shape[0] < 3:
        return

    phones = centroids_df.index.tolist()
    gt_fb = [front_back(p) for p in phones]
    gt_hml = [height(p) for p in phones]

    Z, X_for_metric = _ward_linkage(centroids_df.values, metric)

    sil_curve = {}
    for k in K_RANGE:
        if k > len(phones) - 1:
            continue
        labels = fcluster(Z, k, criterion="maxclust")
        sil_curve[k] = _silhouette(X_for_metric, labels, metric)

    best_k = max(sil_curve, key=lambda k: (sil_curve[k] if not np.isnan(sil_curve[k]) else -1))
    labels_best = fcluster(Z, best_k, criterion="maxclust")
    labels_k2 = fcluster(Z, 2, criterion="maxclust")
    labels_k3 = fcluster(Z, 3, criterion="maxclust")

    ari_rows.append(dict(
        tag=tag, metric=metric,
        n_phonemes=len(phones),
        best_k=best_k,
        silhouette_at_best_k=sil_curve.get(best_k, np.nan),
        silhouette_curve=str({k: round(v, 3) for k, v in sil_curve.items()}),
        ari_front_back_k2=adjusted_rand_score(gt_fb, labels_k2.tolist()),
        ari_high_mid_low_k3=adjusted_rand_score(gt_hml, labels_k3.tolist()),
        ari_front_back_at_best_k=adjusted_rand_score(gt_fb, labels_best.tolist()),
        ari_high_mid_low_at_best_k=adjusted_rand_score(gt_hml, labels_best.tolist()),
    ))

    fig, ax = plt.subplots(figsize=(max(6, len(phones) * 0.5), 5))
    dendrogram(Z, labels=phones, ax=ax, leaf_rotation=45, leaf_font_size=10)
    ax.set_title(f"Dendrogram — {tag} (Ward, k*={best_k})")
    save_fig(fig, out / "figures" / f"9_1_dendro_{tag.replace('/', '_')}.png")


def _vowel_clustering(vowels, layers, out, ari_rows):
    print("\n   9.1 Vowel clustering")

    if F1_LOB in vowels.columns and F2_LOB in vowels.columns:
        ac_cent = (vowels.groupby("phoneme")[[F1_LOB, F2_LOB]]
                   .mean()
                   .dropna())
        _evaluate_vowel_clustering(ac_cent, "euclidean",
                                   "acoustic_vowels", out, ari_rows)

    for lyr in tqdm(layers, desc="    layers"):
        pair = lyr.get("pca_clust")
        if pair is None:
            print(f"    [WARN] {lyr['name']}: no pca_clust, skipping")
            continue
        ids, vecs = pair
        df = merge(vowels, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        nc = df.groupby("phoneme")[pc_cols].mean().dropna()
        _evaluate_vowel_clustering(nc, "cosine",
                                   f"{lyr['name']}_vowels", out, ari_rows)


def _mixed_clustering(ac, layers, out, ari_rows):
    print("\n   9.2 Mixed (consonants + vowels) clustering")

    keep_phonemes = list(set(filter_canonical_vowels(ac, verbose=False)["phoneme"].unique())
                         | set(p for p in TARGET_CONSONANTS
                               if p in ac["phoneme"].values))

    sub = ac[ac["phoneme"].isin(keep_phonemes)].copy()
    print(f"    phonemes used: {sorted(keep_phonemes)}")

    feat_cols = [F1_LOB, F2_LOB]
    if DURATION_COL in sub.columns:
        feat_cols.append(DURATION_COL)
    if SCG_COL in sub.columns:
        feat_cols.append(SCG_COL)

    ac_cent = sub.groupby("phoneme")[feat_cols].mean()
    ac_cent = ac_cent.dropna(subset=[F1_LOB, F2_LOB])
    ac_cent = ac_cent.dropna(axis=1, how="all")
    ac_cent_filled = ac_cent.fillna(ac_cent.median(numeric_only=True))
    non_constant = ac_cent_filled.std(ddof=0) > 0
    ac_cent_filled = ac_cent_filled.loc[:, non_constant]

    print(f"    features used: {ac_cent_filled.columns.tolist()}")

    X_std = (ac_cent_filled - ac_cent_filled.mean()) / ac_cent_filled.std(ddof=0)

    if X_std.isna().any().any():
        print("    [WARN] residual NaN after standardisation, dropping affected rows")
        X_std = X_std.dropna()

    phones = X_std.index.tolist()
    gt_cv = ["vowel" if is_vowel(p) else "consonant" for p in phones]

    Z, _ = _ward_linkage(X_std.values, "euclidean")
    labels_2 = fcluster(Z, 2, criterion="maxclust")
    ari_cv = adjusted_rand_score(gt_cv, labels_2.tolist())
    ari_rows.append(dict(
        tag="acoustic_mixed", metric="euclidean_zscored",
        n_phonemes=len(phones), best_k=2,
        ari_consonant_vowel_k2=ari_cv,
    ))
    print(f"    ARI consonant/vowel acoustic: {ari_cv:.3f}")

    fig, ax = plt.subplots(figsize=(max(6, len(phones) * 0.5), 5))
    dendrogram(Z, labels=phones, ax=ax, leaf_rotation=45, leaf_font_size=10)
    ax.set_title(f"Dendrogram — acoustic mixed (Ward), ARI(C/V)={ari_cv:.3f}")
    save_fig(fig, out / "figures" / "9_2_dendro_acoustic_mixed.png")

    for lyr in layers:
        pair = lyr.get("pca_clust")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(ac[ac["phoneme"].isin(keep_phonemes)], ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        nc = df.groupby("phoneme")[pc_cols].mean().dropna()
        if nc.shape[0] < 3:
            continue

        phones_n = nc.index.tolist()
        gt_cv_n = ["vowel" if is_vowel(p) else "consonant" for p in phones_n]
        Z, _ = _ward_linkage(nc.values, "cosine")
        labels_2 = fcluster(Z, 2, criterion="maxclust")
        ari_cv_n = adjusted_rand_score(gt_cv_n, labels_2.tolist())
        ari_rows.append(dict(
            tag=f"{lyr['name']}_mixed", metric="cosine",
            n_phonemes=len(phones_n), best_k=2,
            ari_consonant_vowel_k2=ari_cv_n,
        ))
        print(f"    ARI consonant/vowel {lyr['name']}: {ari_cv_n:.3f}")

        fig, ax = plt.subplots(figsize=(max(6, len(phones_n) * 0.5), 5))
        dendrogram(Z, labels=phones_n, ax=ax, leaf_rotation=45, leaf_font_size=10)
        ax.set_title(f"Dendrogram — {lyr['name']} mixed (Ward, cosine), ARI(C/V)={ari_cv_n:.3f}")
        save_fig(fig, out / "figures" / f"9_2_dendro_{lyr['name']}_mixed.png")


def _cluster_speakers(spk_features, rep_name, metric, meta_indexed, out, rows):
    if spk_features.shape[0] < 4:
        return
    common = spk_features.index.intersection(meta_indexed.index)
    if len(common) < 4:
        print(f"    [{rep_name}] not enough common speakers ({len(common)})")
        return
    spk_features = spk_features.loc[common]
    meta_aligned = meta_indexed.loc[common]

    Z, _ = _ward_linkage(spk_features.values, metric)
    labels_2 = fcluster(Z, 2, criterion="maxclust")

    for gt_col in ["l1_status", "gender"]:
        gt = meta_aligned[gt_col].values
        if pd.isna(gt).any():
            continue
        ari = adjusted_rand_score(gt, labels_2)
        rows.append(dict(representation=rep_name,
                         metric=metric,
                         n_speakers=len(common),
                         grouping=gt_col,
                         k=2, ari=ari))
        print(f"    Speaker ARI ({rep_name}, {gt_col}): {ari:.3f}")

    labels_spk = [f"{spk}_{meta_aligned.loc[spk, 'l1_status']}_{meta_aligned.loc[spk, 'gender']}"
                  for spk in spk_features.index]
    fig, ax = plt.subplots(figsize=(max(5, len(spk_features) * 0.5), 5))
    dendrogram(Z, labels=labels_spk, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f"Speaker dendrogram — {rep_name} ({metric})")
    save_fig(fig, out / "figures" / f"9_3_speaker_dendro_{rep_name}.png")


def _speaker_clustering(vowels, layers, out, speaker_rows):
    print("\n   9.3 Speaker clustering")

    speaker_meta = (vowels.groupby("speaker_id")
                    .agg(l1_status=("l1_status", "first"),
                         gender=("gender", "first")))

    if F1_LOB in vowels.columns:
        spk_ph = vowels.groupby(["speaker_id", "phoneme"])[[F1_LOB, F2_LOB]].mean()
        spk_wide = spk_ph.unstack("phoneme")
        spk_wide.columns = ["_".join(c) for c in spk_wide.columns]
        spk_wide = spk_wide.dropna()
        _cluster_speakers(spk_wide, "acoustic", "euclidean",
                          speaker_meta, out, speaker_rows)

    for lyr in layers:
        pair = lyr.get("pca_clust")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(vowels, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        spk_ph = df.groupby(["speaker_id", "phoneme"])[pc_cols].mean()
        spk_wide = spk_ph.unstack("phoneme")
        spk_wide.columns = ["_".join(str(c) for c in col) for col in spk_wide.columns]
        spk_wide = spk_wide.dropna()
        _cluster_speakers(spk_wide, lyr["name"], "cosine",
                          speaker_meta, out, speaker_rows)


def clustering(ac, layers, out):
    print("\n=== 9. Hierarchical Clustering ===")
    vowels = filter_canonical_vowels(ac, verbose=False)

    ari_rows = []
    speaker_rows = []

    _vowel_clustering(vowels, layers, out, ari_rows)
    _mixed_clustering(ac, layers, out, ari_rows)
    _speaker_clustering(vowels, layers, out, speaker_rows)

    save_csv(out / "tables" / "9_clustering_phonemes.csv",
             pd.DataFrame(ari_rows), index=False)
    save_csv(out / "tables" / "9_clustering_speakers.csv",
             pd.DataFrame(speaker_rows), index=False)
    print("Clustering done")
