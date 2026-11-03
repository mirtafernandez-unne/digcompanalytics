"""
analysis.py
===========
Empirical evaluation of DIGCOMPAnalytics v2 on four course-cohort datasets.

  A. Corpus description and observability (base + sensitivity configurations)
  B. K-means on the DigComp feature space (k = 2..7)
  C. Baselines with the same k:
       B1  raw (component, event) counts, log-transformed, standardized and
           reduced with PCA to the same dimensionality as the DigComp space
       B2  direct event-to-area mapping (each pair to its highest-weight area,
           weight 1, no subcompetence distribution), same normalization
  D. Bootstrap stability (adjusted Rand index against the full-sample solution)
  E. Cross-cohort transfer within LED (model fitted on 2024 applied to 2025, 2026)
  F. PCA of the DigComp feature space
  G. Sequential patterns over abstracted action trajectories

Usage: python analysis.py <led_pseudo.csv> <itaa_pseudo.csv> <mapping.csv> <out_dir>
"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digcompanalytics as dca

SEED = 2026
N_BOOT = int(os.environ.get("DCA_N_BOOT", 200))
K_RANGE = range(2, 8)
SENS = {"p50/0.60-0.35": (0.50, 0.60, 0.35), "p75/0.70-0.40": (0.75, 0.70, 0.40),
        "p80/0.75-0.45": (0.80, 0.75, 0.45)}

ACTIONS = {
    "VIEW_RESOURCE": {("recurso", "módulo de curso visto"), ("carpeta", "módulo de curso visto"),
                      ("url", "módulo de curso visto"), ("página", "módulo de curso visto")},
    "SUBMIT_ASSIGNMENT": {("tarea", "se ha enviado una entrega")},
    "VIEW_FEEDBACK": {("tarea", "retroalimentación vista")},
    "START_QUIZ": {("cuestionario", "ha comenzado el intento")},
    "SUBMIT_QUIZ": {("cuestionario", "intento enviado")},
    "REVIEW_QUIZ": {("cuestionario", "intento del cuestionario revisado")},
    "VIEW_FORUM": {("foro", "módulo de curso visto"), ("foro", "tema visto")},
}
PATTERNS = {
    "A": ["VIEW_RESOURCE", "SUBMIT_ASSIGNMENT"],
    "C": ["SUBMIT_QUIZ", "REVIEW_QUIZ", "VIEW_RESOURCE"],
    "E": ["START_QUIZ", "SUBMIT_QUIZ", "REVIEW_QUIZ"],
    "F": ["SUBMIT_ASSIGNMENT", "VIEW_FEEDBACK"],
    "G": ["VIEW_RESOURCE", "VIEW_FORUM", "SUBMIT_ASSIGNMENT"],
}


def load(path):
    d = dca.prepare(pd.read_csv(path, dtype=str))
    d["ts"] = pd.to_datetime(d["timestamp"], format="%d/%m/%y, %H:%M:%S")
    return d


# Clustering uses the areas observable in all datasets (sigma-based criterion):
# percentile ranks of very sparse areas behave as near-binary indicators and
# would dominate Euclidean distances.
CORE = ["norm_information", "norm_communication", "norm_problem"]
K_TRANSFER = 3


def features(out):
    return out[CORE].to_numpy() / 100.0, CORE


def kmeans_eval(X, k):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    return km, silhouette_score(X, km.labels_), davies_bouldin_score(X, km.labels_)


def bootstrap_ari(X, k, ref_labels):
    rng = np.random.default_rng(SEED)
    aris = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(X), len(X))
        km = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1e9))).fit(X[idx])
        aris.append(adjusted_rand_score(ref_labels, km.predict(X)))
    return float(np.mean(aris)), float(np.std(aris))


def baseline_b1(st, students, dim):
    counts = (st.groupby(["actor", "key"]).size().unstack(fill_value=0)
              .reindex(students, fill_value=0))
    Z = StandardScaler().fit_transform(np.log1p(counts.to_numpy(dtype=float)))
    return PCA(n_components=dim, random_state=SEED).fit_transform(Z)


def baseline_b2(st, students, mapping):
    beh = mapping[mapping["student_behavior"] == "yes"].copy()
    beh["area"] = beh["subcompetence"].map(dca.SUB_TO_AREA)
    # Single primary subcompetence per event: highest weight; ties are broken deterministically by the
    # lowest subcompetence code (stable sort), so the baseline does not depend on the row order of the file.
    primary = (beh.sort_values(["weight", "subcompetence"], ascending=[False, True], kind="mergesort")
               .drop_duplicates("key")[["key", "area"]])
    j = st.merge(primary, on="key")
    f = j.groupby(["actor", "area"]).size().unstack(fill_value=0).reindex(students, fill_value=0)
    f = f.reindex(columns=["information", "communication", "problem"], fill_value=0)
    norm = f.apply(lambda c: (c.rank(method="max", pct=True)).where(c > 0, 0.0))
    return norm.to_numpy()


def trajectories(st):
    inv = {k: a for a, ks in ACTIONS.items() for k in ks}
    s = st[st["key"].isin(inv)].sort_values("ts")
    s = s.assign(action=s["key"].map(inv))
    seqs = {}
    for actor, grp in s.groupby("actor"):
        acts = grp["action"].tolist()
        collapsed = [a for i, a in enumerate(acts) if i == 0 or a != acts[i - 1]]
        seqs[actor] = collapsed
    return seqs


def has_pattern(seq, pat):
    n = len(pat)
    return any(seq[i:i + n] == pat for i in range(len(seq) - n + 1))


def main(led_path, itaa_path, map_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mapping = dca.load_mapping(map_path)
    led, itaa = load(led_path), load(itaa_path)
    staff_led, staff_itaa = dca.identify_non_students(led), dca.identify_non_students(itaa)
    # LED cohorts: a student belongs to the cohort of the year of their first event in the
    # course, and only that year's events are used. Students who return in later years
    # (re-enrolled) are not counted again, so the three cohorts are disjoint.
    is_st = led["actor"].notna() & ~led["actor"].isin(staff_led) & (led["origin"].str.lower() != "cli")
    cohort = led[is_st].groupby("actor")["ts"].min().dt.year
    led["cohort"] = led["actor"].map(cohort)
    def led_cohort(y):
        return led[(led["ts"].dt.year == y) & ((led["cohort"] == y) | ~is_st)]
    sets = {
        "ITAA 2026": (itaa, staff_itaa),
        "LED 2024": (led_cohort(2024), staff_led),
        "LED 2025": (led_cohort(2025), staff_led),
        "LED 2026": (led_cohort(2026), staff_led),
    }
    R = {"corpus": [], "observability": [], "kmeans": [], "comparison": [], "pca": [],
         "patterns": [], "levels": [], "transfer": []}
    fitted = {}

    for name, (d, staff) in sets.items():
        out, dg = dca.run(d, mapping, dca.Config(), staff=staff)
        out.to_csv(os.path.join(out_dir, f"students_{name.replace(' ', '_')}.csv"))
        st = d[d["actor"].isin(out.index)]
        students = out.index

        R["corpus"].append({"dataset": name, "students": dg["n_students"],
                            "student_events": dg["n_events_students"],
                            "median_events_per_student": float(st.groupby("actor").size().median()),
                            "distinct_pairs": dg["distinct_pairs_students"],
                            "subcompetences_with_signal": 21 - len(dg["subcompetences_without_signal"]),
                            "subcompetences_without_signal": dg["subcompetences_without_signal"],
                            "tau": dg["tau"], "zero_share": dg["zero_share"]})
        R["levels"].append({"dataset": name, **out["level"].value_counts().to_dict()})

        for cname, (q, r, i) in SENS.items():
            o, _ = dca.run(d, mapping, dca.Config(tau_quantile=q, robust=r, inferable=i), staff=staff)
            t = dca.observability_table(o)
            for area, row in t.iterrows():
                R["observability"].append({"dataset": name, "config": cname, "area": area, **row.to_dict()})

        X, cols = features(out)
        best = None
        for k in K_RANGE:
            km, sil, db = kmeans_eval(X, k)
            R["kmeans"].append({"dataset": name, "k": k, "inertia": km.inertia_,
                                "silhouette": sil, "davies_bouldin": db})
            if best is None or sil > best[1]:
                best = (k, sil, db, km)
        k, sil, db, km = best
        fitted[name] = (km, cols)
        ari_m, ari_s = bootstrap_ari(X, k, km.labels_)
        centers = pd.DataFrame(km.cluster_centers_ * 100, columns=cols).round(1)
        sizes = np.bincount(km.labels_)
        R["comparison"].append({"dataset": name, "space": "DigComp (proposed)", "dim": X.shape[1], "k": k,
                                "silhouette": sil, "davies_bouldin": db, "boot_ari_mean": ari_m,
                                "boot_ari_sd": ari_s, "centers": centers.to_dict("records"),
                                "sizes": sizes.tolist()})

        for bname, Xb in [("B1 raw event counts (PCA)", baseline_b1(st, students, X.shape[1])),
                          ("B2 direct area mapping", baseline_b2(st, students, mapping))]:
            kmb, silb, dbb = kmeans_eval(Xb, k)
            am, asd = bootstrap_ari(Xb, k, kmb.labels_)
            R["comparison"].append({"dataset": name, "space": bname, "dim": Xb.shape[1], "k": k,
                                    "silhouette": silb, "davies_bouldin": dbb,
                                    "boot_ari_mean": am, "boot_ari_sd": asd,
                                    "ari_vs_proposed": adjusted_rand_score(km.labels_, kmb.labels_)})

        p = PCA(random_state=SEED).fit(X)
        R["pca"].append({"dataset": name, "features": cols,
                         "explained": p.explained_variance_ratio_.tolist(),
                         "loadings_pc1": p.components_[0].tolist(),
                         "loadings_pc2": p.components_[1].tolist()})

        seqs = trajectories(st)
        n = len(students)
        n_sub = sum(1 for s in seqs.values() if "SUBMIT_ASSIGNMENT" in s)
        n_quiz = sum(1 for s in seqs.values() if "SUBMIT_QUIZ" in s)
        for pname, pat in PATTERNS.items():
            c = sum(has_pattern(s, pat) for s in seqs.values())
            base = n_sub if "SUBMIT_ASSIGNMENT" in pat else (n_quiz if "SUBMIT_QUIZ" in pat else n)
            R["patterns"].append({"dataset": name, "pattern": pname, "sequence": " > ".join(pat),
                                  "students_with_pattern": c, "pct_all": 100 * c / n,
                                  "eligible": base, "pct_eligible": (100 * c / base) if base else None})

    # E. Cross-cohort transfer with a fixed k: model fitted on one cohort applied to another
    Xs = {n: pd.read_csv(os.path.join(out_dir, f"students_{n.replace(' ', '_')}.csv"),
                         index_col=0)[CORE].to_numpy() / 100.0 for n in sets}
    own = {n: KMeans(n_clusters=K_TRANSFER, n_init=20, random_state=SEED).fit(X) for n, X in Xs.items()}
    for src in sets:
        for dst in sets:
            if src != dst:
                R["transfer"].append({"from": src, "to": dst, "k": K_TRANSFER,
                                      "ari": adjusted_rand_score(own[dst].labels_, own[src].predict(Xs[dst]))})
        R["transfer"].append({"from": src, "to": "centers", "k": K_TRANSFER,
                              "centers": (own[src].cluster_centers_ * 100).round(1).tolist(),
                              "sizes": np.bincount(own[src].labels_).tolist()})

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=float)
    return R


if __name__ == "__main__":
    main(*sys.argv[1:5])
