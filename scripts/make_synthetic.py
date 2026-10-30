"""
make_synthetic.py
=================
Generates synthetic Moodle activity logs with the same structure as the logs analyzed in the
article (Spanish-language Moodle interface, standard log export columns).

No real record is used. The generator only reads synthetic_profiles.json, which contains
aggregate figures per dataset: the number of students, the mean and standard deviation of the
log number of events per student, the observation window, and the relative frequency of each
(component, event, origin) combination. Combinations observed fewer than three times were
removed from the profiles.

Each synthetic student receives a total number of events drawn from a log-normal distribution
and a personal mix of event types drawn from a Dirichlet distribution centered on the course mix,
so that students differ in how they distribute their activity across areas. Teaching staff
accounts perform teacher-only actions, so the actor filter can be exercised.

The synthetic logs reproduce the structure of the data and allow the full pipeline to run;
they do not reproduce the numerical results reported in the article.

Usage: python make_synthetic.py [scale] [seed]
    scale  fraction of the original activity volume to generate (default 0.25)
    seed   random seed (default 2026)
Outputs: ../data/synthetic/led_synthetic.csv and ../data/synthetic/itaa_synthetic.csv
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "synthetic")
CONCENTRATION = 15.0
TEACHER_ACTIONS = [("Sistema", "Pregunta creada", "web"), ("Cuestionario", "Informe de cuestionario visto", "web"),
                   ("Informe del calificador", "Informe del calificador visto", "web")]


def random_times(rng, start, end, n):
    t0, t1 = pd.Timestamp(start).value, pd.Timestamp(end).value + 86_399 * 10**9
    return pd.to_datetime(rng.integers(t0, t1, n))


def gen_dataset(rng, p, prefix, course, scale, returning=None):
    mix = pd.DataFrame(p["student_mix"], columns=["component", "event_name", "origin", "n"])
    base = (mix["n"] / mix["n"].sum()).to_numpy()
    n_st = int(p["n_students"])
    rows = []
    actors = [f"{prefix}_s{i:04d}" for i in range(n_st)]
    for a in actors:
        total = max(1, int(round(scale * np.exp(rng.normal(p["log_mean"], p["log_sd"])))))
        w = rng.dirichlet(CONCENTRATION * base + 1e-3)
        idx = rng.choice(len(mix), size=total, p=w)
        sub = mix.iloc[idx][["component", "event_name", "origin"]].copy()
        sub["actor"] = a
        rows.append(sub)
    # teaching staff: their course maintenance plus at least one teacher-only action each
    smix = pd.DataFrame(p["staff_mix"], columns=["component", "event_name", "origin", "n"])
    for j in range(max(1, int(p["n_staff"]))):
        a = f"{prefix}_t{j:02d}"
        k = max(20, int(scale * smix["n"].sum() / max(1, p["n_staff"]) / 10))
        sub = smix.sample(n=k, replace=True, weights="n", random_state=int(rng.integers(1e9)))[["component", "event_name", "origin"]]
        sub = pd.concat([sub, pd.DataFrame([TEACHER_ACTIONS[j % len(TEACHER_ACTIONS)]],
                                           columns=["component", "event_name", "origin"])])
        sub["actor"] = a
        rows.append(sub)
    d = pd.concat(rows, ignore_index=True)
    d["ts"] = random_times(rng, p["start"], p["end"], len(d))
    if returning:  # a few students who reappear in the following year (tests the cohort rule)
        extra = d[d["actor"].isin(returning["actors"])].sample(frac=0.3, random_state=1).copy()
        extra["ts"] = random_times(rng, returning["start"], returning["end"], len(extra))
        d = pd.concat([d, extra], ignore_index=True)
    d["context"] = course
    d["related_user"] = ""
    d["admin_or_system"] = False
    return d


def to_export(d):
    d = d.sort_values("ts", ascending=False)
    d["timestamp"] = d["ts"].dt.strftime("%d/%m/%y, %H:%M:%S")
    return d[["timestamp", "actor", "related_user", "context", "component", "event_name", "origin", "admin_or_system"]]


def main(scale=0.25, seed=2026):
    rng = np.random.default_rng(seed)
    P = json.load(open(os.path.join(HERE, "synthetic_profiles.json"), encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    itaa = gen_dataset(rng, P["ITAA 2026"], "itaa", "Curso: ITAA (synthetic)", 1.0)
    to_export(itaa).to_csv(os.path.join(OUT, "itaa_synthetic.csv"), index=False)
    parts = []
    for y in (2024, 2025, 2026):
        p = P[f"LED {y}"]
        nxt = P.get(f"LED {y + 1}")
        ret = None
        if nxt:
            ret = {"actors": [f"led{y}_s{i:04d}" for i in range(0, p["n_students"], 15)],
                   "start": nxt["start"], "end": nxt["end"]}
        parts.append(gen_dataset(rng, p, f"led{y}", "Curso: LED (synthetic)", scale, ret))
    led = pd.concat(parts, ignore_index=True)
    to_export(led).to_csv(os.path.join(OUT, "led_synthetic.csv"), index=False)
    print(f"ITAA: {len(itaa):,} rows | LED: {len(led):,} rows -> {OUT}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]) if a else 0.25, int(a[1]) if len(a) > 1 else 2026)
