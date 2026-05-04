import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
import statsmodels.formula.api as smf
from tqdm import tqdm
from utils import is_vowel, save_csv
from stats_utils import merge, save_fig, confidence_ellipse, mantel_test


F1_LOB = "F1_mid_lob"
F2_LOB = "F2_mid_lob"
F1_HZ = "F1_mid"
F2_HZ = "F2_mid"


FRENCH_ORAL_VOWELS = {"i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"}
VOWEL_CANONICAL_MAP = {"u:": "u", "aː": "a"}
TARGET_PHON = {"i", "e", "a", "ɑ", "o", "u", "ə"}


def filter_canonical_vowels(df, verbose=False):
    vowels_all = df[df["phoneme"].apply(is_vowel)].copy()
    vowels_all["phoneme"] = vowels_all["phoneme"].replace(VOWEL_CANONICAL_MAP)
    vowels = vowels_all[vowels_all["phoneme"].isin(FRENCH_ORAL_VOWELS)].copy()
    if verbose:
        dropped = len(vowels_all) - len(vowels)
        print(f"  Vowel tokens: {len(vowels)} kept / {dropped} dropped "
              f"({dropped / max(1, len(vowels_all)):.1%})")
        print(f"  Tokens per phoneme:\n{vowels['phoneme'].value_counts().to_string()}")
    return vowels


def desc_acoustic(df, out, intra_spk_phon=TARGET_PHON):
    print("\n=== Acoustic Descriptive Statistics ===")

    vowels = filter_canonical_vowels(df, verbose=True)
    grp_cols = ["phoneme", "l1_status", "gender"]

    rows = []
    for (ph, l1, gen), g in vowels.groupby(grp_cols):
        for feat_lob, feat_hz in [(F1_LOB, F1_HZ), (F2_LOB, F2_HZ)]:
            if feat_lob not in g.columns:
                continue
            v_lob = g[feat_lob].dropna()
            if len(v_lob) == 0:
                continue
            v_hz = g[feat_hz].dropna() if feat_hz in g.columns else pd.Series(dtype=float)
            cv_hz = (v_hz.std() / v_hz.mean()
                     if len(v_hz) > 1 and v_hz.mean() > 0 else np.nan)
            rows.append(dict(
                phoneme=ph, l1_status=l1, gender=gen, feature=feat_lob,
                n=len(v_lob),
                mean=v_lob.mean(), median=v_lob.median(), std=v_lob.std(),
                iqr=v_lob.quantile(0.75) - v_lob.quantile(0.25),
                cv_hz=cv_hz,
            ))
    desc = pd.DataFrame(rows)
    save_csv(out / "tables" / "5_1_acoustic_descriptive.csv", desc, index=True)

    var_rows = []
    for ph, g in vowels.groupby("phoneme"):
        if F1_LOB not in g.columns:
            continue
        g = g[[F1_LOB, "speaker_id", "sentence_id"]].dropna()
        if g["speaker_id"].nunique() < 3 or len(g) < 20:
            continue
        try:
            m = smf.mixedlm(
                f"{F1_LOB} ~1",
                data=g,
                groups=g["speaker_id"],
                re_formula="~1",
                vc_formula={"sentence": "0 + C(sentence_id)"},
            ).fit(reml=True, method="lbfgs")

            if m.cov_re.shape[0] == 0:
                raise ValueError("cov_re is empty")
            var_inter = float(m.cov_re.iloc[0, 0])
            var_intra = float(m.vcomp[0]) if len(m.vcomp) else 0.0
            var_resid = float(m.scale)
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

    save_csv(out / "tables" / "5_1_variance_decomposition.csv",
             pd.DataFrame(var_rows), index=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"L1_f": "#1f77b4", "L1_m": "#aec7e8", "L2_f": "#d62728", "L2_m": "#f7b6d2"}
    markers = {"L1_f": "o", "L1_m": "s", "L2_f": "^", "L2_m": "D"}
    for (l1, gen), g in vowels.groupby(["l1_status", "gender"]):
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
            ax.annotate(ph, (x, -y), fontsize=7, ha="center", va="center")

    patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
    ax.legend(handles=patches, fontsize=8)
    ax.set_xlabel("F2 (Lobanov)")
    ax.set_ylabel("–F1 (Lobanov, IPA convention)")
    ax.set_title("Vowel chart – per-phoneme centroids with 95% ellipses")
    save_fig(fig, out / "figures" / "5_1_vowel_chart.png")

    for feat in [F1_LOB, F2_LOB]:
        if feat not in vowels.columns:
            continue
        fig, ax = plt.subplots(figsize=(15, 5))
        phonemes_sorted = sorted(vowels["phoneme"].unique())
        cells = [("L1", "f"), ("L1", "m"), ("L2", "f"), ("L2", "m")]
        cell_colors = {("L1", "f"): "#1f77b4", ("L1", "m"): "#aec7e8",
                       ("L2", "f"): "#d62728", ("L2", "m"): "#f7b6d2"}
        n_cells = len(cells)
        box_w = 0.18

        bp_handles = {}
        for c_idx, (l1, gen) in enumerate(cells):
            offset = (c_idx - (n_cells - 1) / 2) * box_w
            data, positions = [], []
            for k, ph in enumerate(phonemes_sorted):
                sub = vowels[(vowels["phoneme"] == ph)
                             & (vowels["l1_status"] == l1)
                             & (vowels["gender"] == gen)]
                vals = sub[feat].dropna().values
                data.append(vals if len(vals) > 0 else [np.nan])
                positions.append(k + offset)

            bp = ax.boxplot(data, positions=positions, widths=box_w * 0.85,
                            patch_artist=True,
                            boxprops=dict(facecolor=cell_colors[(l1, gen)], alpha=0.7),
                            medianprops=dict(color="black"),
                            flierprops=dict(marker=".", markersize=2))
            bp_handles[(l1, gen)] = bp["boxes"][0]

        ax.set_xticks(range(len(phonemes_sorted)))
        ax.set_xticklabels(phonemes_sorted, fontsize=9)
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} per phoneme – stratified by L1 × gender")
        ax.legend([bp_handles[c] for c in cells],
                  [f"{l1}/{g}" for l1, g in cells], fontsize=8)
        save_fig(fig, out / "figures" / f"5_1_boxplot_{feat}.png")

    selected = sorted([p for p in intra_spk_phon if p in vowels["phoneme"].values])
    if selected and F1_LOB in vowels.columns:
        fig, axes = plt.subplots(1, len(selected),
                                 figsize=(4 * len(selected), 4), sharey=True)
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
        axes[0].set_ylabel(F1_LOB)
        fig.suptitle("Intra-speaker F1 variability across repetitions")
        save_fig(fig, out / "figures" / "5_1_intra_speaker_strip.png")

    print("\nAcoustic descriptive statistics done")


def desc_neural(meta, layers, out):
    print("\n=== Neural Descriptive Statistics ===")
    meta_vowels = filter_canonical_vowels(meta, verbose=False)

    sep_rows = []

    for lyr in tqdm(layers, desc="Neural stats"):
        name = lyr["name"]
        for rep_key, label in [("pca2", "PCA"), ("umap2", "UMAP")]:
            pair = lyr.get(rep_key)
            if pair is None:
                continue
            ids, vecs = pair

            df_plot = merge(meta, ids, vecs)
            df_metric = merge(meta_vowels, ids, vecs)
            if df_plot.empty:
                continue

            pc_cols = [c for c in df_plot.columns if c.startswith("pc_")]

            for colour_by in ["phoneme", "l1_status", "gender"]:
                if colour_by not in df_plot.columns:
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                cats = df_plot[colour_by].unique()
                cmap = plt.get_cmap("tab20", len(cats))
                for k, cat in enumerate(cats):
                    sub = df_plot[df_plot[colour_by] == cat]
                    ax.scatter(sub[pc_cols[0]], sub[pc_cols[1]],
                               c=[cmap(k)], s=8, alpha=0.5, label=str(cat))
                ax.legend(fontsize=6, markerscale=2,
                          loc="upper right", ncol=2)
                ax.set_title(f"{name} – {label} coloured by {colour_by}")
                ax.set_xlabel(f"{label}1")
                ax.set_ylabel(f"{label}2")
                save_fig(fig, out / "figures" /
                         f"5_2_neural_{name}_{label}_{colour_by}.png")

            if df_metric.empty or len(pc_cols) < 2:
                continue

            ph_means = df_metric.groupby("phoneme")[pc_cols[:2]].mean()
            total_var = df_metric[pc_cols[:2]].var().sum()
            between_var = ph_means.var().sum()
            ratio = between_var / total_var if total_var > 0 else np.nan

            ph_labels = df_metric["phoneme"].values
            X = df_metric[pc_cols].values.astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            Xn = X / norms

            sample_size = min(500, len(X))
            idx = np.random.default_rng(0).choice(len(X), sample_size, replace=False)
            sim_mat = Xn[idx] @ Xn[idx].T
            labels_sub = ph_labels[idx]
            labels_sub = np.asarray(labels_sub)
            same = labels_sub[:, None] == labels_sub[None, :]
            np.fill_diagonal(same, False)
            within = sim_mat[same].mean() if same.any() else np.nan
            between = sim_mat[~same].mean() if (~same).any() else np.nan
            sim_ratio = within / between if (between not in (0,)
                                             and not np.isnan(between)) else np.nan

            sep_rows.append(dict(
                layer=name, dr=label,
                between_class_var_ratio=ratio,
                cosine_within=within,
                cosine_between=between,
                cosine_ratio=sim_ratio,
                n_sample=sample_size,
            ))
            print(f"    {name} {label}: between-class var ratio = {ratio:.3f}, "
                  f"within={within:.3f}, between={between:.3f}, ratio={sim_ratio:.3f}")

    save_csv(out / "tables" / "5_2_neural_class_separation.csv",
             pd.DataFrame(sep_rows), index=True)
    print("\nNeural descriptive statistics done")


def desc_rsm(ac, layers, out):
    print("\n=== Cross-Representation Comparison (Mantel) ===")

    vowels = filter_canonical_vowels(ac, verbose=False)
    phones = sorted(vowels["phoneme"].unique())
    if len(phones) < 3:
        print("Not enough phonemes for RSM")
        return

    if F1_LOB not in vowels.columns or F2_LOB not in vowels.columns:
        print("    F1_lob / F2_lob missing, skipping")
        return

    ac_cent = vowels.groupby("phoneme")[[F1_LOB, F2_LOB]].mean().dropna()
    phones_ac = ac_cent.index.tolist()
    D_ac = cdist(ac_cent.values, ac_cent.values)

    neural_rsms = {}
    for lyr in layers:
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
        if nc.shape[0] < 3:
            continue

        V = nc.values.astype(np.float32)
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        Vn = V / nrm
        D_cos = np.clip(1 - Vn @ Vn.T, 0, None)
        neural_rsms[lyr["name"]] = (nc.index.tolist(), D_cos)

    rows = []
    for name, (phones_n, D_n) in tqdm(neural_rsms.items(), desc="ac vs neural"):
        shared = [p for p in phones_ac if p in phones_n]
        if len(shared) < 3:
            continue
        i_ac = [phones_ac.index(p) for p in shared]
        i_n = [phones_n.index(p)  for p in shared]
        Dac_sub = D_ac[np.ix_(i_ac, i_ac)]
        Dn_sub = D_n[np.ix_(i_n,  i_n)]
        r, p = mantel_test(Dac_sub, Dn_sub, n_perm=5000)
        rows.append(dict(comparison=f"acoustic vs {name}",
                         n_phonemes=len(shared),
                         mantel_r=r, mantel_p=p))
        print(f"    Mantel acoustic vs {name}: r={r:.3f}, p={p:.4f}")

    layer_names = list(neural_rsms.keys())
    for i, n1 in enumerate(layer_names):
        for j, n2 in enumerate(layer_names):
            if j <= i:
                continue
            sh1, D1 = neural_rsms[n1]
            sh2, D2 = neural_rsms[n2]
            shared = [p for p in sh1 if p in sh2]
            if len(shared) < 3:
                continue
            i1 = [sh1.index(p) for p in shared]
            i2 = [sh2.index(p) for p in shared]
            r, p = mantel_test(D1[np.ix_(i1, i1)], D2[np.ix_(i2, i2)],
                               n_perm=5000)
            rows.append(dict(comparison=f"{n1} vs {n2}",
                             n_phonemes=len(shared),
                             mantel_r=r, mantel_p=p))
            print(f"Mantel {n1} vs {n2}: r={r:.3f}, p={p:.4f}")

    save_csv(out / "tables" / "5_3_mantel_rsm.csv",
             pd.DataFrame(rows), index=True)
    print("\nRSM done")
