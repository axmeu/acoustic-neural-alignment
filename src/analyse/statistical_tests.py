
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
from utils import save_csv
from stats_utils import (merge, save_fig, mantel_test, bootstrap_ci,
                         centroid_cosine_dist)
from analyse.descriptive import (F1_LOB, F2_LOB, filter_canonical_vowels,)


def _l1l2_acoustic_tests(vowels, out):
    print("\n   L1 vs L2 acoustic tests")
    phones = sorted(vowels["phoneme"].unique())
    rows = []

    for feat in [F1_LOB, F2_LOB]:
        if feat not in vowels.columns:
            continue
        feat_pvals = []
        feat_phonemes = []

        for ph in phones:
            sub = vowels[vowels["phoneme"] == ph]
            l1_vals = sub[sub["l1_status"] == "L1"][feat].dropna().values
            l2_vals = sub[sub["l1_status"] == "L2"][feat].dropna().values

            if len(l1_vals) < 5 or len(l2_vals) < 5:
                continue

            n_max = 5000
            _, p_norm_l1 = stats.shapiro(l1_vals[:n_max])
            _, p_norm_l2 = stats.shapiro(l2_vals[:n_max])
            _, p_lev = stats.levene(l1_vals, l2_vals)

            if p_norm_l1 > 0.05 and p_norm_l2 > 0.05 and p_lev > 0.05:
                stat_v, p = stats.ttest_ind(l1_vals, l2_vals)
                test = "t-test"
            else:
                stat_v, p = stats.mannwhitneyu(l1_vals, l2_vals,
                                               alternative="two-sided")
                test = "Mann-Whitney"

            rows.append(dict(
                phoneme=ph, feature=feat, test=test, stat=stat_v,
                n_L1=len(l1_vals), n_L2=len(l2_vals),
                mean_L1=l1_vals.mean(), mean_L2=l2_vals.mean(),
                p_raw=p,
            ))
            feat_pvals.append(p)
            feat_phonemes.append(ph)

        if feat_pvals:
            _, p_corr, _, _ = multipletests(feat_pvals, method="fdr_bh")
            for ph, p_bh in zip(feat_phonemes, p_corr):
                for r in rows:
                    if r["phoneme"] == ph and r["feature"] == feat:
                        r["p_bh"] = p_bh
                        break

    save_csv(out / "tables" / "6_1_l1l2_acoustic_tests.csv",
             pd.DataFrame(rows), index=True)


def _gender_residual_test(vowels, out):
    print("\n   Gender residual test (per phoneme)")
    rows = []

    for feat in [F1_LOB, F2_LOB]:
        if feat not in vowels.columns:
            continue
        for ph, g in vowels.groupby("phoneme"):
            spk_means = (g.groupby(["speaker_id", "gender"])[feat]
                         .mean().reset_index())
            f_vals = spk_means[spk_means["gender"] == "f"][feat].values
            m_vals = spk_means[spk_means["gender"] == "m"][feat].values
            if len(f_vals) < 3 or len(m_vals) < 3:
                continue
            stat_v, p = stats.ttest_ind(f_vals, m_vals)
            rows.append(dict(
                phoneme=ph, feature=feat,
                n_f=len(f_vals), n_m=len(m_vals),
                mean_f=f_vals.mean(), mean_m=m_vals.mean(),
                stat=stat_v, p_raw=p,
            ))

        feat_rows = [r for r in rows if r["feature"] == feat]
        if feat_rows:
            pvals = [r["p_raw"] for r in feat_rows]
            _, p_corr, _, _ = multipletests(pvals, method="fdr_bh")
            for r, p_bh in zip(feat_rows, p_corr):
                r["p_bh"] = p_bh

    save_csv(out / "tables" / "6_1_gender_residual.csv",
             pd.DataFrame(rows), index=True)


def _l1l2_neural_permutation(vowels, layers, out, n_perm=5000):
    print(f"\n  L1 vs L2 neural permutation test ({n_perm} perms, speaker-level)")
    rows = []

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

        ph_pvals = []
        ph_names = []

        for ph in sorted(df["phoneme"].unique()):
            sub = df[df["phoneme"] == ph].copy()
            spk_l1 = sub.groupby("speaker_id")["l1_status"].first()
            if spk_l1.value_counts().min() < 2:
                continue

            l1v = sub[sub["l1_status"] == "L1"][pc_cols].values
            l2v = sub[sub["l1_status"] == "L2"][pc_cols].values
            if len(l1v) < 3 or len(l2v) < 3:
                continue

            obs = centroid_cosine_dist(l1v, l2v)

            rng = np.random.default_rng(42)
            speakers = spk_l1.index.values
            labels = spk_l1.values

            count = 0
            for _ in range(n_perm):
                perm_labels = rng.permutation(labels)
                perm_map = dict(zip(speakers, perm_labels))
                perm_l1 = sub["speaker_id"].map(perm_map)
                v1 = sub[perm_l1 == "L1"][pc_cols].values
                v2 = sub[perm_l1 == "L2"][pc_cols].values
                if len(v1) == 0 or len(v2) == 0:
                    continue
                d = centroid_cosine_dist(v1, v2)
                if d >= obs:
                    count += 1

            p = (count + 1) / (n_perm + 1)
            rows.append(dict(layer=lyr["name"], phoneme=ph,
                             obs_dist=obs, p_raw=p,
                             n_L1_speakers=int((labels == "L1").sum()),
                             n_L2_speakers=int((labels == "L2").sum())))
            ph_pvals.append(p)
            ph_names.append(ph)

        if ph_pvals:
            _, p_corr, _, _ = multipletests(ph_pvals, method="fdr_bh")
            for ph, p_bh in zip(ph_names, p_corr):
                for r in rows:
                    if r["layer"] == lyr["name"] and r["phoneme"] == ph:
                        r["p_bh"] = p_bh
                        break

    save_csv(out / "tables" / "6_1_neural_permutation.csv",
             pd.DataFrame(rows), index=True)


def group_comparisons(ac, layers, out):
    print("\n=== Group Comparisons ===")
    vowels = filter_canonical_vowels(ac, verbose=False)
    _l1l2_acoustic_tests(vowels, out)
    _gender_residual_test(vowels, out)
    _l1l2_neural_permutation(vowels, layers, out, n_perm=5000)
    print("Group comparisons done")


def _acoustic_distance_matrices(vowels, phones):
    ac_cent = (vowels.groupby("phoneme")[[F1_LOB, F2_LOB]]
               .mean()
               .reindex(phones)
               .dropna())
    phones = ac_cent.index.tolist()

    D_euc = cdist(ac_cent.values, ac_cent.values, metric="euclidean")

    cov_terms = []
    n_terms = 0
    for ph in phones:
        sub = vowels[vowels["phoneme"] == ph][[F1_LOB, F2_LOB]].dropna()
        if len(sub) >= 3:
            cov_terms.append((len(sub) - 1) * np.cov(sub.values.T))
            n_terms += (len(sub) - 1)

    if cov_terms and n_terms > 0:
        pooled_cov = sum(cov_terms) / n_terms
        try:
            VI = np.linalg.inv(pooled_cov)
            D_mah = cdist(ac_cent.values, ac_cent.values,
                          metric="mahalanobis", VI=VI)
        except np.linalg.LinAlgError:
            print("    [WARN] Mahalanobis: pooled covariance singular, falling back to Euclidean")
            D_mah = D_euc.copy()
    else:
        D_mah = D_euc.copy()

    return phones, D_euc, D_mah


def _neural_distance_matrices(vowels, layers, phones):
    out = {}
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
        nc = (df.groupby("phoneme")[pc_cols].mean()
              .reindex(phones).dropna())
        if nc.shape[0] < 3:
            continue

        V = nc.values.astype(np.float32)
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        Vn = V / nrm
        D_cos = np.clip(1 - Vn @ Vn.T, 0, None)
        out[lyr["name"]] = (nc.index.tolist(), D_cos)
    return out


def _mantel_distance_comparisons(phones, D_euc, D_mah, neural_dists, out,
                                 n_perm=5000):
    print(f"\n  Mantel comparisons ({n_perm} perms)")
    rows = []
    layer_names = list(neural_dists.keys())

    for n1 in tqdm(layer_names, desc="    Mantel ac vs neural"):
        sh1, D1 = neural_dists[n1]
        idx_ac = [phones.index(p) for p in sh1 if p in phones]
        sh1_aligned = [p for p in sh1 if p in phones]
        idx_n = [sh1.index(p) for p in sh1_aligned]
        if len(idx_ac) < 3:
            continue

        Dac_e = D_euc[np.ix_(idx_ac, idx_ac)]
        Dac_m = D_mah[np.ix_(idx_ac, idx_ac)]
        D1_sub = D1[np.ix_(idx_n, idx_n)]

        r, p = mantel_test(Dac_e, D1_sub, n_perm=n_perm)
        rows.append(dict(pair=f"acoustic_euclidean vs {n1}",
                         n_phonemes=len(idx_ac), r=r, p=p))
        r, p = mantel_test(Dac_m, D1_sub, n_perm=n_perm)
        rows.append(dict(pair=f"acoustic_mahalanobis vs {n1}",
                         n_phonemes=len(idx_ac), r=r, p=p))

    for i, n1 in enumerate(layer_names):
        for j, n2 in enumerate(layer_names):
            if j <= i:
                continue
            sh1, D1 = neural_dists[n1]
            sh2, D2 = neural_dists[n2]
            shared = [p for p in sh1 if p in sh2]
            if len(shared) < 3:
                continue
            i1 = [sh1.index(p) for p in shared]
            i2 = [sh2.index(p) for p in shared]
            r, p = mantel_test(D1[np.ix_(i1, i1)], D2[np.ix_(i2, i2)],
                               n_perm=n_perm)
            rows.append(dict(pair=f"{n1} vs {n2}",
                             n_phonemes=len(shared), r=r, p=p))

    save_csv(out / "tables" / "6_2_mantel_distances.csv",
             pd.DataFrame(rows), index=True)


def _bootstrap_pair_distances(vowels, layers, out, n_boot=2000):
    print(f"\n      Bootstrap CIs on selected vowel pairs ({n_boot} resamples)")
    phones = sorted(vowels["phoneme"].unique())

    pair_candidates = [("e", "ɛ"), ("y", "u"), ("i", "e"), ("a", "ɑ"), ("o", "u")]
    selected = [(p1, p2) for (p1, p2) in pair_candidates
                if p1 in phones and p2 in phones]
    print(f"    pairs: {selected}")

    boot_rows = []
    for p1, p2 in selected:
        sub = vowels[vowels["phoneme"].isin([p1, p2])].copy()

        def ac_dist(d, _p1=p1, _p2=p2):
            c1 = d[d["phoneme"] == _p1][[F1_LOB, F2_LOB]].mean().values
            c2 = d[d["phoneme"] == _p2][[F1_LOB, F2_LOB]].mean().values
            if np.isnan(c1).any() or np.isnan(c2).any():
                return np.nan
            return float(np.linalg.norm(c1 - c2))

        ci_lo, ci_hi = bootstrap_ci(sub, ac_dist, n_boot=n_boot, verbose=False)
        boot_rows.append(dict(pair=f"{p1}-{p2}", representation="acoustic",
                              point=ac_dist(sub),
                              ci_lo=ci_lo, ci_hi=ci_hi))

        for lyr in layers:
            pair_data = lyr.get("pca_clust")
            if pair_data is None:
                continue
            ids, vecs = pair_data
            df = merge(vowels, ids, vecs)
            if df.empty:
                continue
            pc_cols = [c for c in df.columns if c.startswith("pc_")]
            dfv = df[df["phoneme"].isin([p1, p2])].copy()
            if dfv.empty:
                continue

            def ne_dist(d, _pc=pc_cols, _p1=p1, _p2=p2):
                v1 = d[d["phoneme"] == _p1][_pc].values
                v2 = d[d["phoneme"] == _p2][_pc].values
                if len(v1) == 0 or len(v2) == 0:
                    return np.nan
                return centroid_cosine_dist(v1, v2)

            ci_lo2, ci_hi2 = bootstrap_ci(dfv, ne_dist,
                                          n_boot=n_boot, verbose=False)
            boot_rows.append(dict(pair=f"{p1}-{p2}",
                                  representation=lyr["name"],
                                  point=ne_dist(dfv),
                                  ci_lo=ci_lo2, ci_hi=ci_hi2))

    save_csv(out / "tables" / "6_2_bootstrap_distances.csv",
             pd.DataFrame(boot_rows), index=True)


def _loso_classifier(vowels, layers, out):
    print("\n   LOSO nearest-centroid classifier")

    def evaluate(df_cls, feat_cols, metric):
        speakers = df_cls["speaker_id"].unique()
        y_true, y_pred, token_ids = [], [], []
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
            cv = centroids.values
            for _, row in test.iterrows():
                v = row[feat_cols].values.astype(np.float32).reshape(1, -1)
                d = cdist(v, cv, metric=metric)[0]
                pred = centroids.index[d.argmin()]
                y_true.append(row["phoneme"])
                y_pred.append(pred)
                token_ids.append(row["phoneme_id"])
        return y_true, y_pred, token_ids

    classif_rows = []
    predictions = {}

    if F1_LOB in vowels.columns:
        df_cls = vowels.dropna(subset=[F1_LOB, F2_LOB])
        y_true, y_pred, tok = evaluate(df_cls, [F1_LOB, F2_LOB], "euclidean")
        if y_true:
            predictions["acoustic"] = (y_true, y_pred, tok)
            classif_rows.append(dict(
                representation="acoustic",
                accuracy=accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, average="macro",
                                  labels=list(set(y_true)), zero_division=0),
                n_tokens=len(y_true),
            ))

    for lyr in layers:
        pair = lyr.get("pca_clust")
        if pair is None:
            continue
        ids, vecs = pair
        df = merge(vowels, ids, vecs)
        if df.empty:
            continue
        pc_cols = [c for c in df.columns if c.startswith("pc_")]
        df_cls = df.dropna(subset=pc_cols)
        y_true, y_pred, tok = evaluate(df_cls, pc_cols, "cosine")
        if y_true:
            predictions[lyr["name"]] = (y_true, y_pred, tok)
            classif_rows.append(dict(
                representation=lyr["name"],
                accuracy=accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, average="macro",
                                  labels=list(set(y_true)), zero_division=0),
                n_tokens=len(y_true),
            ))

    for rep_name, (yt, yp, _) in predictions.items():
        labels_ord = sorted(set(yt))
        cm = confusion_matrix(yt, yp, labels=labels_ord)
        fig, ax = plt.subplots(figsize=(max(6, len(labels_ord) * 0.6),
                                        max(5, len(labels_ord) * 0.5)))
        im = ax.imshow(cm, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(labels_ord)))
        ax.set_yticks(range(len(labels_ord)))
        ax.set_xticklabels(labels_ord, rotation=90, fontsize=8)
        ax.set_yticklabels(labels_ord, fontsize=8)
        plt.colorbar(im, ax=ax)
        acc = accuracy_score(yt, yp)
        ax.set_title(f"Confusion matrix — {rep_name} (acc={acc:.2f})")
        save_fig(fig, out / "figures" / f"6_2_confusion_{rep_name}.png")

    save_csv(out / "tables" / "6_2_classifier_accuracy.csv",
             pd.DataFrame(classif_rows), index=True)

    return predictions


def _mcnemar_classifier_pairs(predictions, out):
    print("\n   McNemar tests across representations")
    rows = []
    rep_names = list(predictions.keys())

    for i, r1 in enumerate(rep_names):
        for j, r2 in enumerate(rep_names):
            if j <= i:
                continue
            yt1, yp1, tok1 = predictions[r1]
            yt2, yp2, tok2 = predictions[r2]

            df1 = pd.DataFrame({"phoneme_id": tok1, "true": yt1, "pred": yp1})
            df2 = pd.DataFrame({"phoneme_id": tok2, "true": yt2, "pred": yp2})
            merged = df1.merge(df2, on="phoneme_id", suffixes=("_1", "_2"))
            if len(merged) < 30:
                continue

            ok1 = (merged["true_1"] == merged["pred_1"]).values
            ok2 = (merged["true_2"] == merged["pred_2"]).values

            table = np.array([
                [int((ok1 & ok2).sum()),  int((ok1 & ~ok2).sum())],
                [int((~ok1 & ok2).sum()), int((~ok1 & ~ok2).sum())],
            ])

            discordant = table[0, 1] + table[1, 0]
            try:
                result = mcnemar(table,
                                 exact=(discordant < 25),
                                 correction=True)
                rows.append(dict(
                    pair=f"{r1} vs {r2}",
                    n_matched=int(len(merged)),
                    acc_1=float(ok1.mean()), acc_2=float(ok2.mean()),
                    discordant_1_only=int(table[0, 1]),
                    discordant_2_only=int(table[1, 0]),
                    statistic=float(result.statistic),
                    p_value=float(result.pvalue),
                ))
            except Exception as exc:
                print(f"    [WARN] McNemar {r1} vs {r2} failed: {exc}")

    save_csv(out / "tables" / "6_2_mcnemar.csv",
             pd.DataFrame(rows), index=True)


def distances(ac, layers, out):
    print("\n=== Inter-phoneme Distances ===")
    vowels = filter_canonical_vowels(ac, verbose=False)
    if F1_LOB not in vowels.columns:
        print("  No normalised acoustics, skipping")
        return

    phones_init = sorted(vowels["phoneme"].unique())
    phones, D_euc, D_mah = _acoustic_distance_matrices(vowels, phones_init)
    neural_dists = _neural_distance_matrices(vowels, layers, phones)

    _mantel_distance_comparisons(phones, D_euc, D_mah, neural_dists, out)
    _bootstrap_pair_distances(vowels, layers, out, n_boot=2000)
    predictions = _loso_classifier(vowels, layers, out)
    _mcnemar_classifier_pairs(predictions, out)
    print("Inter-phoneme distances done")
