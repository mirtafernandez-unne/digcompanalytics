"""
scenario_navigation.py
======================
Sensitivity of the results to the navigation rule (Section III-C).

The base mapping assigns navigation events (course page, sections, entry pages of activities)
only to subcompetence 1.1. This scenario removes the rule: the entry page of each activity type
is mapped like the activity itself, and viewing the submission status counts as reviewing one's
work (5.4). Everything else is unchanged.

Usage: python scenario_navigation.py <led.csv> <itaa.csv> <mapping.csv> <out.json>
"""
import sys, os, json
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digcompanalytics as dca
import analysis as an
import robustness as rb

ALTERNATIVE = {
    ("cuestionario", "módulo de curso visto"): [("5.1", .5), ("5.4", .5)],
    ("tarea", "módulo de curso visto"): [("3.1", .6), ("5.2", .4)],
    ("foro", "módulo de curso visto"): [("2.1", .6), ("1.1", .4)],
    ("paquete scorm", "módulo de curso visto"): [("5.2", .6), ("1.1", .4)],
    ("selección de grupo", "módulo de curso visto"): [("2.4", .7), ("2.1", .3)],
    ("tarea", "se ha visualizado el estado de la entrega"): [("5.4", 1.0)],
}


def alternative_mapping(path, out_path):
    raw = pd.read_csv(path)
    for (c, e), assign in ALTERNATIVE.items():
        mask = (raw["component"] == c) & (raw["event_name"] == e)
        if not mask.any():
            raise KeyError((c, e))
        rows = [{"component": c, "event_name": e, "subcompetence": s, "weight": w, "student_behavior": "yes"}
                for s, w in assign]
        raw = pd.concat([raw[~mask], pd.DataFrame(rows)], ignore_index=True)
    raw.to_csv(out_path, index=False)
    return dca.load_mapping(out_path)


def main(led_path, itaa_path, map_path, out_path):
    base = dca.load_mapping(map_path)
    alt = alternative_mapping(map_path, os.path.splitext(out_path)[0] + "_mapping.csv")
    led, itaa = an.load(led_path), an.load(itaa_path)
    R = []
    for name, (d, staff) in rb.datasets(led, itaa).items():
        o0, _ = dca.run(d, base, dca.Config(), staff=staff)
        o1, _ = dca.run(d, alt, dca.Config(), staff=staff)
        t0, t1 = dca.observability_table(o0)["robust"], dca.observability_table(o1)["robust"]
        order = lambda t: t.sort_values(ascending=False, kind="mergesort").index.tolist()
        k = 3 if name.startswith("ITAA") else 2
        ari = adjusted_rand_score(rb.km(o0[an.CORE].to_numpy() / 100, k).labels_,
                                  rb.km(o1[an.CORE].to_numpy() / 100, k).labels_)
        R.append({"dataset": name, "robust_base": t0.to_dict(), "robust_alt": t1.to_dict(),
                  "same_order": order(t0) == order(t1), "k": k, "ari": float(ari),
                  "rho_inf_problem_base": float(spearmanr(o0.norm_information, o0.norm_problem).correlation),
                  "rho_inf_problem_alt": float(spearmanr(o1.norm_information, o1.norm_problem).correlation)})
    json.dump(R, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
