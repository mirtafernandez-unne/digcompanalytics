"""
robustness.py
=============
Robustness analyses requested in the pre-submission review.

  R1. Stability of the fixed k = 3 typology (bootstrap ARI, proposed space and baselines)
  R2. Sensitivity of the clustering to the selection of areas (3 core areas vs 4 in LED)
  R3. Sensitivity to the mapping weights (random perturbation of every event distribution)
  R4. Discriminant evidence: relation of each area with total platform activity

Usage: python robustness.py <led_pseudo.csv> <itaa_pseudo.csv> <mapping.csv> <out.json>
"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digcompanalytics as dca
import analysis as an

SEED = 2026
N_PERT = int(os.environ.get("DCA_N_PERT", 500))
CONC = {"moderate": 20.0, "strong": 5.0}   # Dirichlet concentration around the original weights


def datasets(led, itaa):
    sl, si = dca.identify_non_students(led), dca.identify_non_students(itaa)
    is_st = led["actor"].notna() & ~led["actor"].isin(sl) & (led["origin"].str.lower() != "cli")
    coh = led[is_st].groupby("actor")["ts"].min().dt.year
    led = led.assign(cohort=led["actor"].map(coh))
    sets = {"ITAA 2026": (itaa, si)}
    for y in (2024, 2025, 2026):
        sets[f"LED {y}"] = (led[(led["ts"].dt.year == y) & ((led["cohort"] == y) | ~is_st)], sl)
    return sets


def km(X, k, seed=SEED, n_init=20):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def perturb(mapping, conc, rng):
    out = []
    for key, g in mapping.groupby("key", sort=False):
        g = g.copy()
        if len(g) > 1:
            g["weight"] = rng.dirichlet(conc * g["weight"].to_numpy())
        out.append(g)
    return pd.concat(out)


G = None


def perturb_job(args):
    level, conc, name, li = args
    mapping, sets, base = G
    d, staff = sets[name]
    rng = np.random.default_rng([SEED, li, list(sets).index(name)])
    b = base[name]
    bt = dca.observability_table(b)["robust"]
    b_order = bt.sort_values(ascending=False, kind="mergesort").index.tolist()
    bX = b[an.CORE].to_numpy() / 100.0
    b_lab = {k: km(bX, k).labels_ for k in (2, 3)}
    same_order, rhos, aris2, aris3, rob_shift = 0, [], [], [], []
    for _ in range(N_PERT):
        pm = perturb(mapping, conc, rng)
        o, _ = dca.run(d, pm, dca.Config(), staff=staff)
        t = dca.observability_table(o)["robust"]
        same_order += int(t.sort_values(ascending=False, kind="mergesort").index.tolist() == b_order)
        rob_shift.append(float((t - bt).abs().max()))
        rhos.append(min(spearmanr(o[c], b[c]).correlation for c in an.CORE))
        pX = o[an.CORE].to_numpy() / 100.0
        aris2.append(adjusted_rand_score(b_lab[2], km(pX, 2, n_init=5).labels_))
        aris3.append(adjusted_rand_score(b_lab[3], km(pX, 3, n_init=5).labels_))
    print(level, name, "done", flush=True)
    return {"level": level, "dataset": name, "n": N_PERT,
            "same_area_order_pct": 100 * same_order / N_PERT,
            "max_robust_shift_pp_mean": float(np.mean(rob_shift)),
            "max_robust_shift_pp_p95": float(np.percentile(rob_shift, 95)),
            "min_rho_mean": float(np.mean(rhos)), "min_rho_p5": float(np.percentile(rhos, 5)),
            "ari_k2_mean": float(np.mean(aris2)), "ari_k2_p5": float(np.percentile(aris2, 5)),
            "ari_k3_mean": float(np.mean(aris3)), "ari_k3_p5": float(np.percentile(aris3, 5))}


def main(led_path, itaa_path, map_path, out_path):
    mapping = dca.load_mapping(map_path)
    led, itaa = an.load(led_path), an.load(itaa_path)
    sets = datasets(led, itaa)
    R = {"k3": [], "areas": [], "weights": [], "discriminant": []}
    base = {}
    for name, (d, staff) in sets.items():
        out, _ = dca.run(d, mapping, dca.Config(), staff=staff)
        base[name] = out
        st = d[d["actor"].isin(out.index) & (d["origin"].str.lower() != "cli")]
        X = out[an.CORE].to_numpy() / 100.0
        # R1: k = 3 stability for the three spaces
        for label, Xs in [("Proposed", X), ("B1", an.baseline_b1(st, out.index, 3)), ("B2", an.baseline_b2(st, out.index, mapping))]:
            ref = km(Xs, 3)
            m_, s_ = an.bootstrap_ari(Xs, 3, ref.labels_)
            from sklearn.metrics import silhouette_score
            R["k3"].append({"dataset": name, "space": label, "silhouette": float(silhouette_score(Xs, ref.labels_)),
                            "boot_ari_mean": m_, "boot_ari_sd": s_})
        # R2: 3 core areas vs adding content creation (LED only, where content is observable)
        if name.startswith("LED"):
            cols4 = an.CORE + ["norm_content"]
            X4 = out[cols4].to_numpy() / 100.0
            for k in (2, 3):
                a3, a4 = km(X, k).labels_, km(X4, k).labels_
                R["areas"].append({"dataset": name, "k": k, "ari_3_vs_4": float(adjusted_rand_score(a3, a4))})
        # R4: relation with total activity (all behavioral student events)
        beh = set(mapping.loc[mapping["student_behavior"] == "yes", "key"])
        total = st[st["key"].isin(beh)].groupby("actor").size().reindex(out.index, fill_value=0)
        row = {"dataset": name}
        for a in ["information", "communication", "content", "problem"]:
            f = out[f"freq_{a}"]
            row[a] = float(spearmanr(f, total).correlation) if f.std() > 0 else None
        R["discriminant"].append(row)

    json.dump(R, open(out_path, "w"), indent=1)   # partial save (R1, R2, R4)
    # R3: weight perturbation, one job per (level, dataset)
    global G
    G = (mapping, sets, base)
    jobs = [(lvl, c, n, i) for i, (lvl, c) in enumerate(CONC.items()) for n in sets]
    R["weights"] = [perturb_job(j) for j in jobs]   # sequential: fork + sklearn threads can deadlock
    json.dump(R, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
