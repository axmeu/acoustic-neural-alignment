from utils import is_vowel, nfc, save_csv, front_back, height
from stats_utils import (load_layer,
                         merge,
                         save_fig,
                         mantel_test,
                         bootstrap_ci,
                         centroid_cosine_dist,
                         confidence_ellipse
                         )
import argparse
import warnings
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                             f1_score, accuracy_score, silhouette_score)
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings("ignore")
matplotlib.use("Agg")


F1 = "F1_mid"
F2 = "F2_mid"
F3 = "F3_mid"
F0 = "f0_mid"
F1_LOB = "F1_mid_lob"
F2_LOB = "F2_mid_lob"
F1_25 = "F1_25"
F2_25 = "F2_25"
F1_75 = "F1_75"
F2_75 = "F2_75"

TARGET_PHON = {"ə", "u", "o", "e", "ɑ",
               "a", "i"}

###########################
# 5 Descriptive Statistics
###########################


def desc_acoustic(df, out, intra_spk_phon=TARGET_PHON, ):
    print("\n=== Acoustic Descriptive Statistics ===")
    grp_cols = ["phoneme", "l1_status", "gender"]
    vowels = df[df["phoneme"].apply(is_vowel)]

    # ── Summary table ──
    rows = []
    for (ph, l1, gen), g in vowels.groupby(grp_cols):
        for col in [F1_LOB, F2_LOB]:
            if col not in g.columns:
                continue
            v = g[col].dropna()
            rows.append(dict(phoneme=ph, l1_status=l1, gender=gen, feature=col,
                             mean=v.mean(), median=v.median(), std=v.std(),
                             iqr=v.quantile(0.75) - v.quantile(0.25),
                             cv=v.std() / abs(v.mean()) if v.mean() != 0 else np.nan))
    desc = pd.DataFrame(rows)
    save_csv(out / "tables" / "acoustic_descriptive.csv", desc, index=True)

    # ── Variance decomposition for F1 ──
    var_rows = []
    for ph, g in vowels.groupby("phoneme"):
        if F1_LOB not in g.columns:
            continue
        g = g[[F1_LOB, "speaker_id", "sentence_id"]].dropna()
        if g["speaker_id"].nunique() < 3 or len(g) < 20:
            continue
        try:
            # Random intercept for speaker + variance component for sentence (within speaker)
            m = smf.mixedlm(
                f"{F1_LOB} ~ 1",
                data=g,
                groups=g["speaker_id"],
                vc_formula={"sentence": "0 + C(sentence_id)"},
            ).fit(reml=True, method="lbfgs")

            var_inter = float(m.cov_re.iloc[0, 0])           # σ²_speaker
            var_intra = float(m.vcomp[0]) if len(m.vcomp) else 0.0  # σ²_sentence|speaker
            var_resid = float(m.scale)                        # σ²_residual
            var_total = var_inter + var_intra + var_resid

            var_rows.append(dict(
                phoneme=ph,
                var_total=var_total,
                var_inter_speaker=var_inter,
                var_intra_speaker=var_intra,
                var_residual=var_resid,
                prop_inter=var_inter / var_total,
                prop_intra=var_intra / var_total,
                prop_resid=var_resid / var_total,
                n_tokens=len(g),
                n_speakers=g["speaker_id"].nunique(),
            ))
        except Exception as exc:
            print(f"    Variance decomposition failed for /{ph}/: {exc}")

    save_csv(out / "tables" / "5_desc_variance_decomposition.csv",
             pd.DataFrame(var_rows), index=True)

    # ── Vowel chart (F1 vs F2, IPA convention: F1 inverted) ──
    fig, ax = plt.subplots(figsize=(9, 7))
    groups = vowels.groupby(["l1_status", "gender"])
    colors = {"L1_f": "#1f77b4", "L1_m": "#aec7e8", "L2_f": "#d62728", "L2_m": "#f7b6d2"}
    markers = {"L1_f": "o", "L1_m": "s", "L2_f": "^", "L2_m": "D"}
    for (l1, gen), g in groups:
        label = f"{l1}_{gen}"
        col = colors.get(label, "grey")
        mk = markers.get(label, "o")
        for ph, pg in g.groupby("phoneme"):
            x = pg[F2_LOB].mean()
            y = pg[F1_LOB].mean()
            ax.scatter(x, -y, c=col, marker=mk, s=40, alpha=0.5)
            confidence_ellipse(pg[F2_LOB].dropna().values,
                               -pg[F1_LOB].dropna().values,
                               ax, edgecolor=col, alpha=0.4)
        centroid_x = g.groupby("phoneme")[F2_LOB].mean()
        centroid_y = g.groupby("phoneme")[F1_LOB].mean()
        for ph in centroid_x.index:
            ax.annotate(ph, (centroid_x[ph], -centroid_y[ph]),
                        fontsize=7, ha="center", va="center")

    patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
    ax.legend(handles=patches, fontsize=8)
    ax.set_xlabel("F2 (Lobanov)")
    ax.set_ylabel("–F1 (Lobanov, IPA convention)")
    ax.set_title("Vowel chart – per-phoneme centroids with 95% ellipses")
    save_fig(fig, out / "figures" / "desc_vowel_chart.png")

    # ── Box plots F1 / F2 per phoneme by L1 status ──
    for feat in [F1_LOB, F2_LOB]:
        if feat not in vowels.columns:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        phonemes_sorted = sorted(vowels["phoneme"].unique())
        positions_l1, positions_l2 = [], []
        data_l1, data_l2 = [], []
        w = 0.35
        for k, ph in enumerate(phonemes_sorted):
            sub = vowels[vowels["phoneme"] == ph]
            data_l1.append(sub[sub["l1_status"] == "L1"][feat].dropna().values)
            data_l2.append(sub[sub["l1_status"] == "L2"][feat].dropna().values)
            positions_l1.append(k - w / 2)
            positions_l2.append(k + w / 2)
        bp1 = ax.boxplot(data_l1, positions=positions_l1, widths=w * 0.8,
                         patch_artist=True,
                         boxprops=dict(facecolor="#aec7e8"),
                         medianprops=dict(color="navy"),
                         flierprops=dict(marker=".", markersize=2))
        bp2 = ax.boxplot(data_l2, positions=positions_l2, widths=w * 0.8,
                         patch_artist=True,
                         boxprops=dict(facecolor="#f7b6d2"),
                         medianprops=dict(color="darkred"),
                         flierprops=dict(marker=".", markersize=2))
        ax.set_xticks(range(len(phonemes_sorted)))
        ax.set_xticklabels(phonemes_sorted, fontsize=7)
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} per phoneme – L1 (blue) vs L2 (red)")
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["L1", "L2"])
        save_fig(fig, out / "figures" / f"5_boxplot_{feat}.png")

    # ── Intra-speaker variability strip plot for selected vowels ──
    selected = [nfc(p) for p in intra_spk_phon if nfc(p) in vowels["phoneme"].values]
    if selected and F1_LOB in vowels.columns:
        fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4), sharey=True)
        if len(selected) == 1:
            axes = [axes]
        for ax, ph in zip(axes, selected):
            sub = vowels[vowels["phoneme"] == ph]
            for spk, sg in sub.groupby("speaker_id"):
                l1 = sg["l1_status"].iloc[0]
                c = "#1f77b4" if l1 == "L1" else "#d62728"
                x_vals = sg["repetition"].values if "repetition" in sg.columns \
                    else np.zeros(len(sg))
                ax.scatter(x_vals, sg[F1_LOB].values, c=c, alpha=0.4, s=15)
            ax.set_title(f"/{ph}/")
            ax.set_xlabel("Repetition")
        axes[0].set_ylabel(f"{F1_LOB}")
        fig.suptitle("Intra-speaker F1 variability across repetitions")
        save_fig(fig, out / "figures" / "5_intra_speaker_strip.png")

    print("\nAcoustic descriptive statistics script done")


def desc_neural(meta, layers, out):
    print("\n=== Neural Descriptive Statistics ===")
    for lyr in tqdm(layers, desc="Neural stats"):
        name = lyr["name"]
        for rep_key, label in [("pca2", "PCA"), ("umap2", "UMAP")]:
            pair = lyr.get(rep_key)
            if pair is None:
                continue
            ids, vecs = pair
            df = merge(meta, ids, vecs)
            if df.empty:
                continue

            pc_cols = [c for c in df.columns if c.startswith("pc_")]

            # ── 2D projections coloured by phoneme / L1 / gender ──
            for colour_by in ["phoneme", "l1_status", "gender"]:
                if colour_by not in df.columns:
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                cats = df[colour_by].unique()
                cmap = plt.get_cmap("tab20", len(cats))
                for k, cat in enumerate(cats):
                    sub = df[df[colour_by] == cat]
                    ax.scatter(sub[pc_cols[0]], sub[pc_cols[1]],
                               c=[cmap(k)], s=8, alpha=0.5, label=str(cat))
                ax.legend(fontsize=6, markerscale=2,
                          loc="upper right", ncol=2)
                ax.set_title(f"{name} – {label} coloured by {colour_by}")
                ax.set_xlabel(f"{label}1")
                ax.set_ylabel(f"{label}2")
                save_fig(fig, out / "figures" /
                         f"5_neural_{name}_{label}_{colour_by}.png")

            # ── Between-class variance ratio ──
            if len(pc_cols) >= 2:
                ph_means = df.groupby("phoneme")[pc_cols[:2]].mean()
                total_var = df[pc_cols[:2]].var().sum()
                between_var = ph_means.var().sum()
                ratio = between_var / total_var if total_var > 0 else np.nan
                print(f"    {name} {label}: between-class var ratio = {ratio:.3f}")

            # ── Cosine similarity within vs between phoneme ──
            if len(pc_cols) >= 2:
                ph_labels = df["phoneme"].values
                vecs2 = df[pc_cols].values.astype(np.float32)
                norms = np.linalg.norm(vecs2, axis=1, keepdims=True)
                norms[norms == 0] = 1
                vecs_n = vecs2 / norms
                idx = np.random.default_rng(0).choice(len(vecs2),
                                                      min(500, len(vecs2)),
                                                      replace=False)
                sim_mat = vecs_n[idx] @ vecs_n[idx].T
                labels_sub = ph_labels[idx]
                labels_sub = np.array(ph_labels[idx])
                same_mask = labels_sub[:, None] == labels_sub[None, :]
                np.fill_diagonal(same_mask, False)
                within = sim_mat[same_mask].mean() if same_mask.any() else np.nan
                between = sim_mat[~same_mask].mean() if (~same_mask).any() else np.nan
                print(f"    {name} {label}: within={within:.3f}, "
                      f"between={between:.3f}, ratio={within/between:.3f}")

    print("\nNeural descriptive statistics script done")


def desc_rsm(ac, layers, out):
    print("\n=== Cross-Representation Comparison (Mantel) ===")
    vowels = ac[ac["phoneme"].apply(is_vowel)]
    phones = sorted(vowels["phoneme"].unique())
    if len(phones) < 3:
        print("Not enough phonemes for RSM")
        return

    # Acoustic RSM
    if F1_LOB in vowels.columns and F2_LOB in vowels.columns:
        ac_cent = vowels.groupby("phoneme")[[F1_LOB, F2_LOB]].mean()
        ac_cent = ac_cent.reindex(phones).dropna()
        phones_ac = ac_cent.index.tolist()
        D_ac = cdist(ac_cent.values, ac_cent.values)
    else:
        phones_ac, D_ac = [], None

    rows = []
    for lyr in tqdm(layers, desc="RSM"):
        pair = lyr.get("pca_clust") or lyr.get("pca2")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(ac, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        dfv = df[df["phoneme"].apply(is_vowel)]
        neural_cent = dfv.groupby("phoneme")[pc_cols].mean()
        neural_cent = neural_cent.reindex(phones_ac).dropna()
        if neural_cent.shape[0] < 3:
            continue

        # cosine distance matrix
        vn = neural_cent.values.astype(np.float32)
        nrm = np.linalg.norm(vn, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        D_neural = 1 - (vn / nrm) @ (vn / nrm).T
        D_neural = np.clip(D_neural, 0, None)

        # align phoneme sets
        shared = [p for p in phones_ac if p in neural_cent.index]
        if len(shared) < 3:
            continue
        idx_ac = [phones_ac.index(p) for p in shared]
        idx_ne = [list(neural_cent.index).index(p) for p in shared]
        Dac_sub = D_ac[np.ix_(idx_ac, idx_ac)]
        Dne_sub = D_neural[np.ix_(idx_ne, idx_ne)]

        r, p = mantel_test(Dac_sub, Dne_sub)
        rows.append(dict(layer=lyr["name"], mantel_r=r, mantel_p=p))
        print(f"    Mantel: acoustic vs {lyr['name']}: r={r:.3f}, p={p:.4f}")

    save_csv(out / "tables" / "5_mantel_rsm.csv", pd.DataFrame(rows), index=True)
    print("\nRSM done")


###################
# Statistical Tests
###################

def group_comparisons(ac, layers: list[dict], out: Path):
    print("\n=== Group Comparisons ===")
    vowels = ac[ac["phoneme"].apply(is_vowel)]
    phones = sorted(vowels["phoneme"].unique())
    rows = []

    for feat in [F1_LOB, F2_LOB]:
        if feat not in vowels.columns:
            continue
        pvals = []
        for ph in phones:
            sub = vowels[vowels["phoneme"] == ph]
            l1_vals = sub[sub["l1_status"] == "L1"][feat].dropna().values
            l2_vals = sub[sub["l1_status"] == "L2"][feat].dropna().values
            if len(l1_vals) < 5 or len(l2_vals) < 5:
                pvals.append(np.nan)
                continue
            # normality
            _, p_norm_l1 = stats.shapiro(l1_vals[:50])
            _, p_norm_l2 = stats.shapiro(l2_vals[:50])
            # levene
            _, p_lev = stats.levene(l1_vals, l2_vals)
            if p_norm_l1 > 0.05 and p_norm_l2 > 0.05 and p_lev > 0.05:
                stat, p = stats.ttest_ind(l1_vals, l2_vals)
                test = "t-test"
            else:
                stat, p = stats.mannwhitneyu(l1_vals, l2_vals, alternative="two-sided")
                test = "Mann-Whitney"
            pvals.append(p)
            rows.append(dict(phoneme=ph, feature=feat, test=test,
                             stat=stat, p_raw=p,
                             mean_L1=l1_vals.mean(), mean_L2=l2_vals.mean()))

        # BH correction
        valid = [(i, p) for i, p in enumerate(pvals) if not np.isnan(p)]
        if valid:
            idx_v, pv = zip(*valid)
            _, p_corr, _, _ = multipletests(list(pv), method="fdr_bh")
            for k, i in enumerate(idx_v):
                rows_i = [r for r in rows if r["phoneme"] == phones[i]
                          and r["feature"] == feat]
                if rows_i:
                    rows_i[-1]["p_bh"] = p_corr[k]

    save_csv(out / "tables" / "6_l1l2_acoustic_tests.csv", pd.DataFrame(rows), index=True)

    # Gender test (after Lobanov normalisation, residual effect)
    gender_rows = []
    for feat in [F1_LOB, F2_LOB]:
        if feat not in vowels.columns:
            continue
        spk_means = vowels.groupby(["speaker_id", "gender"])[feat].mean().reset_index()
        f = spk_means[spk_means["gender"] == "f"][feat].values
        m = spk_means[spk_means["gender"] == "m"][feat].values
        if len(f) >= 3 and len(m) >= 3:
            stat, p = stats.ttest_ind(f, m)
            gender_rows.append(dict(feature=feat, stat=stat, p=p))
    save_csv(out / "tables" / "6_gender_residual.csv", pd.DataFrame(gender_rows), index=True)

    # Neural permutation test
    print("Running neural permutation tests...")
    perm_rows = []
    for lyr in tqdm(layers, desc="Neural permutation"):
        pair = lyr.get("pca_clust") or lyr.get("pca2")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(ac, ids, vecs)
        if df.empty or "l1_status" not in df.columns:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        dfv = df[df["phoneme"].apply(is_vowel)]
        ph_pvals = []
        ph_phones = sorted(dfv["phoneme"].unique())
        rng = np.random.default_rng(42)
        for ph in ph_phones:
            sub = dfv[dfv["phoneme"] == ph]
            l1v = sub[sub["l1_status"] == "L1"][pc_cols].values
            l2v = sub[sub["l1_status"] == "L2"][pc_cols].values
            if len(l1v) < 3 or len(l2v) < 3:
                ph_pvals.append(np.nan)
                continue
            obs = centroid_cosine_dist(l1v, l2v)
            all_v = np.vstack([l1v, l2v])
            n1 = len(l1v)
            count = 0
            N_PERM = 5000
            for _ in range(N_PERM):
                perm = rng.permutation(len(all_v))
                d = centroid_cosine_dist(all_v[perm[:n1]], all_v[perm[n1:]])
                if d >= obs:
                    count += 1
            p = (count + 1) / (N_PERM + 1)
            ph_pvals.append(p)
            perm_rows.append(dict(layer=lyr["name"], phoneme=ph,
                                  obs_dist=obs, p_raw=p))

        # BH
        valid = [(i, p) for i, p in enumerate(ph_pvals) if not np.isnan(p)]
        if valid:
            idx_v, pv = zip(*valid)
            _, p_corr, _, _ = multipletests(list(pv), method="fdr_bh")
            for k, i in enumerate(idx_v):
                rr = [r for r in perm_rows
                      if r["layer"] == lyr["name"] and r["phoneme"] == ph_phones[i]]
                if rr:
                    rr[-1]["p_bh"] = p_corr[k]

    save_csv(out / "tables" / "6_neural_permutation.csv", pd.DataFrame(perm_rows), index=True)
    print("Group comparison done")


def distances(ac, layers, out):
    print("\n=== Inter-phoneme Distances ===")
    vowels = ac[ac["phoneme"].apply(is_vowel)]
    phones = sorted(vowels["phoneme"].unique())

    if F1_LOB not in vowels.columns:
        print("  No normalised acoustics – skipping.")
        return

    ac_cent = vowels.groupby("phoneme")[[F1_LOB, F2_LOB]].mean().reindex(phones)
    ac_cent = ac_cent.dropna()
    phones = ac_cent.index.tolist()

    # Euclidean
    D_euc = cdist(ac_cent.values, ac_cent.values)
    # Mahalanobis
    cov_parts = []
    for ph in phones:
        sub = vowels[vowels["phoneme"] == ph][[F1_LOB, F2_LOB]].dropna()
        if len(sub) >= 3:
            cov_parts.append(np.cov(sub.values.T) * (len(sub) - 1))
    if cov_parts:
        pooled_cov = sum(cov_parts) / len(cov_parts)
        try:
            VI = np.linalg.inv(pooled_cov)
            D_mah = cdist(ac_cent.values, ac_cent.values, metric="mahalanobis", VI=VI)
        except np.linalg.LinAlgError:
            D_mah = D_euc.copy()
    else:
        D_mah = D_euc.copy()

    # Neural distance matrices
    neural_dists = {}
    for lyr in tqdm(layers, desc="Distances"):
        pair = lyr.get("pca_clust") or lyr.get("pca2")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(ac, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        dfv = df[df["phoneme"].apply(is_vowel)]
        nc = dfv.groupby("phoneme")[pc_cols].mean().reindex(phones).dropna()
        shared = nc.index.tolist()
        if len(shared) < 3:
            continue
        vn = nc.values.astype(np.float32)
        nrm = np.linalg.norm(vn, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        D_neural = np.clip(1 - (vn / nrm) @ (vn / nrm).T, 0, None)
        neural_dists[lyr["name"]] = (shared, D_neural)

    # Mantel comparisons
    print("\nMantel comparisons...")
    mantel_rows = []
    layer_names = list(neural_dists.keys())
    for i, n1 in tqdm(list(enumerate(layer_names)),
                      total=len(layer_names),
                      desc="Mantel"):
        sh1, D1 = neural_dists[n1]
        idx_ac = [phones.index(p) for p in sh1 if p in phones]
        Dac_sub = D_euc[np.ix_(idx_ac, idx_ac)]
        r, p = mantel_test(Dac_sub, D1)
        mantel_rows.append(dict(pair=f"acoustic_euc vs {n1}", r=r, p=p))
        r, p = mantel_test(D_mah[np.ix_(idx_ac, idx_ac)], D1)
        mantel_rows.append(dict(pair=f"acoustic_mah vs {n1}", r=r, p=p))
        for j, n2 in enumerate(layer_names):
            if j <= i:
                continue
            sh2, D2 = neural_dists[n2]
            shared12 = [p for p in sh1 if p in sh2]
            if len(shared12) < 3:
                continue
            i1 = [sh1.index(p) for p in shared12]
            i2 = [sh2.index(p) for p in shared12]
            r, p = mantel_test(D1[np.ix_(i1, i1)], D2[np.ix_(i2, i2)])
            mantel_rows.append(dict(pair=f"{n1} vs {n2}", r=r, p=p))

    save_csv(out / "tables" / "6_mantel_distances.csv", pd.DataFrame(mantel_rows), index=True)

    # Bootstrap CIs on selected pairs
    selected_pairs = []
    pair_candidates = [("e", "ɛ"), ("o", "ɔ"), ("y", "u"), ("i", "e"), ("a", "ɑ")]
    for p1, p2 in pair_candidates:
        p1n, p2n = nfc(p1), nfc(p2)
        if p1n in phones and p2n in phones:
            selected_pairs.append((p1n, p2n))

    boot_rows = []
    for p1, p2 in selected_pairs:
        i1, i2 = phones.index(p1), phones.index(p2)
        # acoustic bootstrap
        sub = vowels[vowels["phoneme"].isin([p1, p2])].copy()

        def ac_dist(df):
            c1 = df[df["phoneme"] == p1][[F1_LOB, F2_LOB]].mean().values
            c2 = df[df["phoneme"] == p2][[F1_LOB, F2_LOB]].mean().values
            return float(np.linalg.norm(c1 - c2))
        ci_lo, ci_hi = bootstrap_ci(sub, ac_dist, n_boot=2000)
        boot_rows.append(dict(pair=f"{p1}-{p2}", rep="acoustic",
                              point=ac_dist(sub), ci_lo=ci_lo, ci_hi=ci_hi))
        # neural bootstrap
        for lyr in layers:
            pair = lyr.get("pca_clust") or lyr.get("pca2")
            if pair is None:
                continue
            ids, vecs = pair
            df = merge(ac, ids, vecs)
            if df.empty:
                continue
            pc_cols = [c for c in df.columns if c.startswith("pc_")]
            dfv = df[df["phoneme"].isin([p1, p2])]
            if dfv.empty:
                continue

            def ne_dist(d, _pc=pc_cols, _p1=p1, _p2=p2):
                v1 = d[d["phoneme"] == _p1][_pc].values
                v2 = d[d["phoneme"] == _p2][_pc].values
                if len(v1) == 0 or len(v2) == 0:
                    return np.nan
                return centroid_cosine_dist(v1, v2)
            ci_lo2, ci_hi2 = bootstrap_ci(dfv, ne_dist, n_boot=2000)
            boot_rows.append(dict(pair=f"{p1}-{p2}", rep=lyr["name"],
                                  point=ne_dist(dfv), ci_lo=ci_lo2, ci_hi=ci_hi2))

    save_csv(out / "tables" / "6_bootstrap_distances.csv", pd.DataFrame(boot_rows), index=True)

    # Nearest-centroid classifier (leave-one-speaker-out)
    classif_rows = []
    for rep_name, get_data in (
        [("acoustic", None)] +
        [(lyr["name"], lyr) for lyr in layers]
    ):
        if rep_name == "acoustic":
            if F1_LOB not in vowels.columns:
                continue
            feat_cols = [F1_LOB, F2_LOB]
            df_cls = vowels.dropna(subset=feat_cols)
        else:
            pair = get_data.get("pca_clust") or get_data.get("pca2")
            if pair is None:
                continue
            ids, vecs = pair
            df_cls = merge(ac, ids, vecs)
            if df_cls.empty:
                continue
            df_cls = df_cls[df_cls["phoneme"].apply(is_vowel)]
            feat_cols = [c for c in df_cls.columns if c.startswith("pc_")]

        if "speaker_id" not in df_cls.columns:
            continue
        speakers = df_cls["speaker_id"].unique()
        y_true, y_pred = [], []

        for spk in speakers:
            train = df_cls[df_cls["speaker_id"] != spk]
            test = df_cls[df_cls["speaker_id"] == spk]

            if train.empty or test.empty:
                continue

            centroids = (train.groupby("phoneme")[feat_cols]
                         .mean(numeric_only=True)
                         .astype(np.float32))

            if centroids.empty:
                continue

            centroid_values = centroids.values.astype(np.float32)

            for _, row in test.iterrows():
                v = row[feat_cols].values.astype(np.float32).reshape(1, -1)

                dists = cdist(v, centroid_values)[0]
                pred = centroids.index[dists.argmin()]

                y_true.append(row["phoneme"])
                y_pred.append(pred)

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro",
                          labels=list(set(y_true)), zero_division=0)
            classif_rows.append(dict(representation=rep_name,
                                     accuracy=acc, macro_f1=f1,
                                     n_tokens=len(y_true)))
            # Confusion matrix
            labels_ord = sorted(set(y_true))
            cm = confusion_matrix(y_true, y_pred, labels=labels_ord)
            fig, ax = plt.subplots(figsize=(max(6, len(labels_ord) * 0.6),
                                            max(5, len(labels_ord) * 0.5)))
            im = ax.imshow(cm, aspect="auto", cmap="Blues")
            ax.set_xticks(range(len(labels_ord)))
            ax.set_yticks(range(len(labels_ord)))
            ax.set_xticklabels(labels_ord, rotation=90, fontsize=7)
            ax.set_yticklabels(labels_ord, fontsize=7)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Confusion matrix – {rep_name} (acc={acc:.2f})")
            save_fig(fig, out / "figures" / f"6_confusion_{rep_name}.png")

    save_csv(out / "tables" / "6_classifier_accuracy.csv",
             pd.DataFrame(classif_rows), index=True)
    print("Group comparison done")


#############################
# Linear Mixed-Effects Models
#############################

def lme(ac, layers, out):
    print("\n=== Linear Mixed-Effects Models ===")
    vowels = ac[ac["phoneme"].apply(is_vowel)].copy()

    if F1_LOB not in vowels.columns:
        print("  F1_mid_lob missing – skipping LME.")
        return

    # Encode fixed effects
    vowels["is_L2"] = (vowels["l1_status"] == "L2").astype(int)
    vowels["is_male"] = (vowels["gender"].str.lower() == "m").astype(int)

    # Diagnostic
    print(f"  Gender values: {vowels['gender'].value_counts().to_dict()}")
    print(f"  is_male sum: {vowels['is_male'].sum()} / {len(vowels)}")
    print(f"  is_L2 sum: {vowels['is_L2'].sum()} / {len(vowels)}")

    lme_rows = []

    target_phonemes = [nfc(p) for p in TARGET_PHON]
    target_phonemes = [p for p in target_phonemes if p in vowels["phoneme"].values]
    print(f"  Target phonemes for LME: {target_phonemes}")

    def fit_models(df, response, tag):
        df = df[[response, "is_L2", "is_male", "speaker_id"]].dropna()
        if df["speaker_id"].nunique() < 4 or len(df) < 20:
            return
        formulas = {
            "null":        f"{response} ~ 1",
            "main":        f"{response} ~ is_L2 + is_male",
            "interaction": f"{response} ~ is_L2 * is_male",
        }
        fitted = {}
        for name, formula in formulas.items():
            try:
                # Use REML for null (ICC), ML for nested LRT comparisons
                use_reml = (name == "null")
                model = smf.mixedlm(formula, df,
                                    groups=df["speaker_id"]).fit(
                                        reml=use_reml, method="lbfgs")
                fitted[name] = model

                if name == "null":
                    var_u = float(model.cov_re.iloc[0, 0])
                    var_e = float(model.scale)
                    icc = var_u / (var_u + var_e) if (var_u + var_e) > 0 else np.nan
                    lme_rows.append(dict(tag=tag, model=name,
                                         aic=model.aic, bic=model.bic,
                                         var_u=var_u, var_e=var_e, icc=icc))
                    print(f"    {tag} null ICC = {icc:.3f}")
                else:
                    lme_rows.append(dict(tag=tag, model=name,
                                         aic=model.aic, bic=model.bic,
                                         icc=np.nan))
            except Exception as exc:
                print(f"    LME failed for {tag}/{name}: {exc}")

        # LRT main vs interaction (both fitted with ML)
        if "main" in fitted and "interaction" in fitted:
            lr = 2 * (fitted["interaction"].llf - fitted["main"].llf)
            p_lr = stats.chi2.sf(lr, df=1)
            lme_rows.append(dict(tag=tag, model="LRT_main_vs_interaction",
                                 lr_stat=lr, lr_p=p_lr,
                                 aic=np.nan, bic=np.nan, icc=np.nan))
            print(f"    {tag} LRT main vs interaction: LR={lr:.2f}, p={p_lr:.4f}")

        # R² (Nakagawa & Schielzeth) — fixed-only prediction
        if "interaction" in fitted:
            m = fitted["interaction"]
            try:
                beta = m.fe_params.values
                X = m.model.exog
                var_f = float(np.var(X @ beta))
                var_u = float(m.cov_re.iloc[0, 0])
                var_e = float(m.scale)
                denom = var_f + var_u + var_e
                r2_marginal = var_f / denom if denom > 0 else np.nan
                r2_conditional = (var_f + var_u) / denom if denom > 0 else np.nan
                lme_rows.append(dict(tag=tag, model="R2",
                                     r2_marginal=r2_marginal,
                                     r2_conditional=r2_conditional,
                                     aic=np.nan, bic=np.nan, icc=np.nan))
                print(f"    {tag}: R²m={r2_marginal:.3f}, R²c={r2_conditional:.3f}")
            except Exception as exc:
                print(f"    R² calc failed for {tag}: {exc}")

    # ── Acoustic models per phoneme ──
    for ph in target_phonemes:
        sub = vowels[vowels["phoneme"] == ph]
        fit_models(sub, F1_LOB, f"acoustic_F1_{ph}")
        fit_models(sub, F2_LOB, f"acoustic_F2_{ph}")

    # ── Neural models per phoneme, on first 5 PCs ──
    for lyr in tqdm(layers, desc="LME neural"):
        pair = lyr.get("pca_lme")
        if pair is None:
            pair = lyr.get("pca_clust") or lyr.get("pca2")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(ac, ids, vecs)
        if df.empty:
            continue
        df["is_L2"] = (df["l1_status"] == "L2").astype(int)
        df["is_male"] = (df["gender"].str.lower() == "m").astype(int)
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        dfv = df[df["phoneme"].apply(is_vowel)]

        for ph in target_phonemes:
            sub = dfv[dfv["phoneme"] == ph]
            if len(sub) < 20:
                continue
            for dim in pc_cols[:5]:
                fit_models(sub, dim, f"{lyr['name']}_{dim}_{ph}")

    save_csv(out / "tables" / "7_lme_results.csv",
             pd.DataFrame(lme_rows), index=True)
    print("LME done")


##################################
# Confidence intervals and ROPE
##################################

def rope(ac, layers, out):
    print("\n=== CIs and ROPE ===")
    vowels = ac[ac["phoneme"].apply(is_vowel)].copy()
    phones = sorted(vowels["phoneme"].unique())

    # ── Acoustic CIs ──
    ac_rows = []
    for ph in phones:
        for feat, rope_lo, rope_hi in [
            (F1_LOB, None, None),      # normalised – ROPE N/A
            (F2_LOB, None, None),
            (F1,     -20.0, 20.0),     # raw Hz – JND ROPE
            (F2,     -20.0, 20.0),
        ]:
            if feat not in vowels.columns:
                continue
            sub = vowels[vowels["phoneme"] == ph]
            l1v = sub[sub["l1_status"] == "L1"][feat].dropna().values
            l2v = sub[sub["l1_status"] == "L2"][feat].dropna().values
            if len(l1v) < 3 or len(l2v) < 3:
                continue
            diff = l1v.mean() - l2v.mean()
            se = np.sqrt(l1v.var() / len(l1v) + l2v.var() / len(l2v))
            t_crit = stats.t.ppf(0.975, df=len(l1v) + len(l2v) - 2)
            ci_lo_val = diff - t_crit * se
            ci_hi_val = diff + t_crit * se
            if rope_lo is not None:
                if ci_hi_val < rope_lo or ci_lo_val > rope_hi:
                    rope_class = "non-equivalent"
                elif ci_lo_val >= rope_lo and ci_hi_val <= rope_hi:
                    rope_class = "equivalent"
                else:
                    rope_class = "indeterminate"
            else:
                rope_class = "N/A (normalised)"
            ac_rows.append(dict(phoneme=ph, feature=feat, point=diff,
                                ci_lo=ci_lo_val, ci_hi=ci_hi_val,
                                rope_lo=rope_lo, rope_hi=rope_hi,
                                rope_class=rope_class))

    save_csv(out / "tables" / "8_acoustic_ci_rope.csv", pd.DataFrame(ac_rows), index=True)

    # ── Forest plot acoustic ──
    def _make_forest_plot(rows, feat_label, fig_path, with_rope=False):
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.4)))
        colors_rope = {"equivalent": "green", "non-equivalent": "red",
                    "indeterminate": "orange", "N/A (normalised)": "grey"}
        for k, r in enumerate(rows):
            c = colors_rope.get(r["rope_class"], "grey")
            ax.barh(k, r["ci_hi"] - r["ci_lo"],
                    left=r["ci_lo"], height=0.4, color=c, alpha=0.5)
            ax.plot([r["point"]], [k], "k|", markersize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        if with_rope and rows[0]["rope_lo"] is not None:
            ax.axvspan(rows[0]["rope_lo"], rows[0]["rope_hi"],
                       alpha=0.1, color="blue",
                       label=f"ROPE [{rows[0]['rope_lo']:.0f}, {rows[0]['rope_hi']:.0f}] Hz")
            ax.legend(loc="lower right")

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["phoneme"] for r in rows], fontsize=8)
        ax.set_xlabel(f"L1 – L2 difference ({feat_label})")
        ax.set_title(f"Forest plot – L1/L2 contrast in {feat_label}")
        save_fig(fig, fig_path)


    # F1 — Hz (with ROPE)
    rows_f1_hz = [r for r in ac_rows if r["feature"] == F1]
    _make_forest_plot(rows_f1_hz, "F1 (Hz)",
                      out / "figures" / "8_forest_acoustic_F1_Hz.png",
                      with_rope=True)

    # F1 — Lobanov (no ROPE)
    rows_f1_lob = [r for r in ac_rows if r["feature"] == F1_LOB]
    _make_forest_plot(rows_f1_lob, "F1 (Lobanov)",
                      out / "figures" / "8_forest_acoustic_F1_Lobanov.png",
                      with_rope=False)

    # F2 — Hz (with ROPE)
    rows_f2_hz = [r for r in ac_rows if r["feature"] == F2]
    _make_forest_plot(rows_f2_hz, "F2 (Hz)",
                      out / "figures" / "8_forest_acoustic_F2_Hz.png",
                      with_rope=True)

    # F2 — Lobanov (no ROPE)
    rows_f2_lob = [r for r in ac_rows if r["feature"] == F2_LOB]
    _make_forest_plot(rows_f2_lob, "F2 (Lobanov)",
                      out / "figures" / "8_forest_acoustic_F2_Lobanov.png",
                      with_rope=False)

    # ── Neural CIs (bootstrap on cosine distance) ──
    neural_rows = []
    for lyr in tqdm(layers, desc="ROPE neural"):
        pair = lyr.get("pca_clust") or lyr.get("pca2")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(ac, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        dfv = df[df["phoneme"].apply(is_vowel)]

        # ROPE_neural: mean intra-speaker cosine distance
        intra_dists = []
        for spk, sg in dfv.groupby("speaker_id"):
            for ph, pg in sg.groupby("phoneme"):
                if len(pg) >= 2:
                    vv = pg[pc_cols].values.astype(np.float32)
                    nrm = np.linalg.norm(vv, axis=1, keepdims=True)
                    nrm[nrm == 0] = 1
                    vn = vv / nrm
                    cos_sim = vn @ vn.T
                    cos_dist = 1 - cos_sim[np.triu_indices(len(vn), k=1)]
                    intra_dists.extend(cos_dist.tolist())
        rope_neural = float(np.nanmean(intra_dists)) if intra_dists else 0.05

        for ph in sorted(dfv["phoneme"].unique()):
            sub = dfv[dfv["phoneme"] == ph]
            l1v = sub[sub["l1_status"] == "L1"][pc_cols].values
            l2v = sub[sub["l1_status"] == "L2"][pc_cols].values
            if len(l1v) < 2 or len(l2v) < 2:
                continue
            point = centroid_cosine_dist(l1v, l2v)

            def _ne_dist(d, _pc=pc_cols):
                v1 = d[d["l1_status"] == "L1"][_pc].values
                v2 = d[d["l1_status"] == "L2"][_pc].values
                if len(v1) == 0 or len(v2) == 0:
                    return np.nan
                return centroid_cosine_dist(v1, v2)

            ci_lo, ci_hi = bootstrap_ci(sub, _ne_dist, n_boot=2000)

            if ci_lo > rope_neural:
                rope_class = "non-equivalent"
            elif ci_hi <= rope_neural:
                rope_class = "equivalent"
            else:
                rope_class = "indeterminate"

            neural_rows.append(dict(layer=lyr["name"], phoneme=ph,
                                    point=point, ci_lo=ci_lo, ci_hi=ci_hi,
                                    rope_delta=rope_neural,
                                    rope_class=rope_class))

    save_csv(out / "tables" / "8_neural_ci_rope.csv", pd.DataFrame(neural_rows), index=True)

    # Forest plot neural (first layer only)
    if neural_rows:
        first_layer = neural_rows[0]["layer"]
        subset = [r for r in neural_rows if r["layer"] == first_layer]
        fig, ax = plt.subplots(figsize=(8, max(4, len(subset) * 0.4)))
        colors_rope = {"equivalent": "green", "non-equivalent": "red",
                       "indeterminate": "orange"}
        for k, r in enumerate(subset):
            c = colors_rope.get(r["rope_class"], "grey")
            ax.barh(k, r["ci_hi"] - r["ci_lo"],
                    left=r["ci_lo"], height=0.4, color=c, alpha=0.5)
            ax.plot([r["point"]], [k], "k|", markersize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        if subset:
            ax.axvline(subset[0]["rope_delta"], color="blue",
                       linestyle=":", label=f"ROPE δ₀={subset[0]['rope_delta']:.3f}")
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels([r["phoneme"] for r in subset], fontsize=8)
        ax.set_xlabel("Cosine distance (L1 vs L2 centroid)")
        ax.set_title(f"Forest plot – neural L1/L2 contrast ({first_layer})")
        ax.legend()
        save_fig(fig, out / "figures" / f"8_forest_neural_{first_layer}.png")

    # ── ROPE summary table ──
    rope_summary = []
    for r in ac_rows:
        if r["feature"] in [F1, F2]:
            rope_summary.append(dict(phoneme=r["phoneme"],
                                     representation=f"acoustic_{r['feature']}",
                                     point=r["point"],
                                     ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                     rope_class=r["rope_class"]))
    for r in neural_rows:
        rope_summary.append(dict(phoneme=r["phoneme"],
                                 representation=r["layer"],
                                 point=r["point"],
                                 ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                 rope_class=r["rope_class"]))
    save_csv(out / "tables" / "8_rope_summary.csv", pd.DataFrame(rope_summary), index=True)
    print("Rope done")


#############
# Clustering
#############

def ari_report(labels_true, labels_pred, tag):
    ari = adjusted_rand_score(labels_true, labels_pred)
    print(f"ARI ({tag}): {ari:.3f}")
    return ari


def clustering(ac, layers, out):
    print("\n=== Hierarchical Clustering ===")
    vowels = ac[ac["phoneme"].apply(is_vowel)].copy()
    phones_all = sorted(vowels["phoneme"].unique())

    # Ground-truth labels per phoneme
    gt_fb = [front_back(p) for p in phones_all]
    gt_hml = [height(p) for p in phones_all]

    ari_rows = []

    def cluster_and_eval(cent_df, metric, tag, phones):
        if cent_df.shape[0] < 3:
            return 0
        idx = [phones_all.index(p) for p in phones if p in phones_all]
        gt_fb_sub = [gt_fb[i] for i in idx]
        gt_hml_sub = [gt_hml[i] for i in idx]

        if metric == "euclidean":
            X = cent_df.values
        else:
            X = cent_df.values.astype(np.float32)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)

        Z = linkage(X, method="ward")
        best_k, best_sil = 2, -1
        for k in range(2, min(len(phones) - 1, 7)):
            labels = fcluster(Z, k, criterion="maxclust")
            try:
                s = silhouette_score(X, labels,
                                     metric="euclidean" if metric == "euclidean" else "cosine")
                if s > best_sil:
                    best_sil, best_k = s, k
            except Exception:
                pass

        labels_k = fcluster(Z, best_k, criterion="maxclust")
        ari_fb = ari_report(gt_fb_sub,  labels_k.tolist(), f"{tag} front/back")
        ari_hml = ari_report(gt_hml_sub, labels_k.tolist(), f"{tag} high/mid/low")
        ari_rows.append(dict(tag=tag, k=best_k, sil=best_sil,
                             ari_front_back=ari_fb, ari_high_mid_low=ari_hml))

        # Dendrogram
        fig, ax = plt.subplots(figsize=(max(6, len(phones) * 0.5), 5))
        dendrogram(Z, labels=phones, ax=ax, leaf_rotation=45, leaf_font_size=9)
        ax.set_title(f"Dendrogram – {tag} (Ward, k*={best_k})")
        save_fig(fig, out / "figures" / f"9_dendro_{tag.replace('/', '_')}.png")
        return best_k

    # ── Acoustic vowels ──
    if F1_LOB in vowels.columns and F2_LOB in vowels.columns:
        ac_cent = vowels.groupby("phoneme")[[F1_LOB, F2_LOB]].mean().reindex(phones_all).dropna()
        cluster_and_eval(ac_cent, "euclidean", "acoustic_vowels", ac_cent.index.tolist())

    # ── Neural vowels ──
    for lyr in tqdm(layers, desc="Clustering vowels"):
        pair = lyr.get("pca_clust") or lyr.get("pca2")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(ac, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        dfv = df[df["phoneme"].apply(is_vowel)]
        nc = dfv.groupby("phoneme")[pc_cols].mean().reindex(phones_all).dropna()
        cluster_and_eval(nc, "cosine", f"{lyr['name']}_vowels", nc.index.tolist())

    # ── Consonants + vowels (acoustic) ──
    if F1_LOB in ac.columns:
        mixed_cent = ac.groupby("phoneme")[[F1_LOB, F2_LOB]].mean().dropna()
        if "duration_ms" in ac.columns:
            dur_cent = ac.groupby("phoneme")["duration_ms"].mean()
            mixed_cent = mixed_cent.join(dur_cent, how="inner")
        phones_mixed = mixed_cent.index.tolist()
        gt_cv = ["vowel" if is_vowel(p) else "consonant" for p in phones_mixed]
        Z_mixed = linkage(mixed_cent.values, method="ward")
        labels_2 = fcluster(Z_mixed, 2, criterion="maxclust")
        ari_cv = ari_report(gt_cv, labels_2.tolist(), "acoustic consonant/vowel")
        ari_rows.append(dict(tag="acoustic_consonants_vowels", k=2,
                             ari_consonant_vowel=ari_cv))

    # ── Speaker clustering ──
    speaker_ari_rows = []
    speaker_meta = ac.groupby("speaker_id").agg(
        l1_status=("l1_status", "first"),
        gender=("gender", "first")
    ).reset_index()

    def speaker_cluster(rep_name, spk_features):
        if spk_features.shape[0] < 4:
            return

        meta_indexed = speaker_meta.set_index("speaker_id")
        common = spk_features.index.intersection(meta_indexed.index)
        if len(common) < 4:
            print(f"speaker_cluster {rep_name}: not enough common speakers ({len(common)})")
            return

        spk_features = spk_features.loc[common]
        meta_aligned = meta_indexed.loc[common]

        Z = linkage(spk_features.values, method="ward")

        # k=2 for L1/L2 and gender
        for k, gt_col in [(2, "l1_status"), (2, "gender")]:
            labels_k = fcluster(Z, k, criterion="maxclust")
            gt = meta_aligned[gt_col].values

            # Skip if ground truth has NaN
            if pd.isna(gt).any():
                print(f"speaker_cluster {rep_name}: NaN in ground truth for {gt_col}, skipping")
                continue

            ari = adjusted_rand_score(gt, labels_k)
            speaker_ari_rows.append(dict(representation=rep_name,
                                         grouping=gt_col, ari=ari))
            print(f"Speaker ARI ({rep_name}, {gt_col}): {ari:.3f}")

        # Dendrogram
        fig, ax = plt.subplots(figsize=(max(5, len(spk_features) * 0.5), 5))
        labels_spk = [f"{spk}_{meta_aligned.loc[spk, 'l1_status']}_{meta_aligned.loc[spk, 'gender']}"
                      for spk in spk_features.index]
        dendrogram(Z, labels=labels_spk, ax=ax, leaf_rotation=90, leaf_font_size=7)
        ax.set_title(f"Speaker dendrogram – {rep_name}")
        save_fig(fig, out / "figures" / f"9_speaker_dendro_{rep_name}.png")

    # Acoustic speaker rep: concatenate per-phoneme F1/F2 means
    if F1_LOB in vowels.columns:
        spk_ph = vowels.groupby(["speaker_id", "phoneme"])[[F1_LOB, F2_LOB]].mean()
        spk_ph_wide = spk_ph.unstack("phoneme")
        spk_ph_wide.columns = ["_".join(c) for c in spk_ph_wide.columns]
        spk_ph_wide = spk_ph_wide.dropna()
        if spk_ph_wide.shape[0] >= 4:
            speaker_cluster("acoustic", spk_ph_wide)

    # Neural speaker rep
    for lyr in tqdm(layers, desc="Clustering speakers"):
        pair = lyr.get("pca_clust") or lyr.get("pca2")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(ac, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        dfv = df[df["phoneme"].apply(is_vowel)]
        spk_ph = dfv.groupby(["speaker_id", "phoneme"])[pc_cols].mean()
        spk_ph_wide = spk_ph.unstack("phoneme")
        spk_ph_wide.columns = ["_".join(str(c) for c in col)
                               for col in spk_ph_wide.columns]
        spk_ph_wide = spk_ph_wide.dropna()
        if spk_ph_wide.shape[0] >= 4:
            speaker_cluster(lyr["name"], spk_ph_wide)

    save_csv(out / "tables" / "9_ari_phonemes.csv", pd.DataFrame(ari_rows), index=True)
    save_csv(out / "tables" / "9_ari_speakers.csv", pd.DataFrame(speaker_ari_rows), index=True)
    print("Clustering done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic", default="outputs/features_acoustic_norm.csv")
    parser.add_argument("--neural-dirs", default="outputs/neural_norm")
    parser.add_argument("--output", default="outputs/analysis")
    args = parser.parse_args()

    out = Path(args.output)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # ── Load acoustic features ──
    ac_path = Path(args.acoustic)
    if not ac_path.exists():
        raise FileNotFoundError(f"Acoustic features not found: {ac_path}")

    ac = pd.read_csv(ac_path)
    # Normalise phoneme strings
    ac["phoneme"] = ac["phoneme"].apply(nfc)
    print(f"Loaded acoustic: {len(ac)} tokens, "
          f"{ac['phoneme'].nunique()} phonemes, "
          f"{ac['speaker_id'].nunique() if 'speaker_id' in ac.columns else '?'} speakers")

    layers = []
    root = Path(args.neural_dirs)

    for d in root.iterdir():
        if not d.is_dir():
            continue
        lyr = load_layer(d)
        layers.append(lyr)

        keys_present = [k for k, v in lyr.items()
                        if k != "name" and v is not None]
        print(f"Loaded layer {lyr['name']}: {keys_present}")

    # ── Run all sections ──
    desc_acoustic(ac, out)
    desc_neural(ac, layers, out)
    desc_rsm(ac, layers, out)

    group_comparisons(ac, layers, out)
    distances(ac, layers, out)

    lme(ac, layers, out)

    rope(ac, layers, out)

    clustering(ac, layers, out)

    print("\nAnalysis complete. Results written to:", out)


if __name__ == "__main__":
    main()
