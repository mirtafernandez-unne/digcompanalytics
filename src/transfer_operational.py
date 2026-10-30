"""
transfer_operational.py
=======================
Operational transfer test requested in the second pre-submission review.

The recurrence test in analysis.py normalizes the target cohort within itself. Here the
target students are instead expressed through the empirical distribution of the SOURCE
cohort (no information from the target distribution is used), assigned to the nearest
source centroid, and compared with the target's own k = 3 partition (ARI).

Usage: python transfer_operational.py <results_dir> <out.json>
"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

SEED, K = 2026, 3
AREAS = ["information", "communication", "problem"]
NAMES = ["ITAA 2026", "LED 2024", "LED 2025", "LED 2026"]


def source_ecdf(src_freq, x):
    """Percentile of x within the source distribution, using the framework's rule:
    share of source students with a value <= x; 0 when there is no activity."""
    s = np.sort(src_freq)
    p = np.searchsorted(s, x, side="right") / len(s)
    return np.where(x > 0, p, 0.0)


def main(res_dir, out_path):
    D = {n: pd.read_csv(os.path.join(res_dir, f"students_{n.replace(' ', '_')}.csv"), index_col=0) for n in NAMES}
    own = {n: KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(d[[f"norm_{a}" for a in AREAS]].to_numpy() / 100)
           for n, d in D.items()}
    out = []
    for src in NAMES:
        for dst in NAMES:
            if src == dst:
                continue
            Xrel = D[dst][[f"norm_{a}" for a in AREAS]].to_numpy() / 100
            Xop = np.column_stack([source_ecdf(D[src][f"freq_{a}"].to_numpy(), D[dst][f"freq_{a}"].to_numpy())
                                   for a in AREAS])
            ref = own[dst].labels_
            out.append({"from": src, "to": dst,
                        "ari_relative": float(adjusted_rand_score(ref, own[src].predict(Xrel))),
                        "ari_operational": float(adjusted_rand_score(ref, own[src].predict(Xop)))})
    json.dump(out, open(out_path, "w"), indent=1)
    for r in out:
        print(f"{r['from']:>9} -> {r['to']:<9} relative {r['ari_relative']:.2f}  operational {r['ari_operational']:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
