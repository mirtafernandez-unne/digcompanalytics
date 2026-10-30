"""
digcompanalytics.py  (v2)
=========================
DIGCOMPAnalytics: DigComp 2.2-guided behavioral proxy inference from Moodle logs.

Changes with respect to v1 (pipeline.py + digcomp_heuristica.py):
  1. The event mapping g: E -> Delta(S) lives in an editable CSV file
     (mapping_digcomp22.csv) instead of being hard-coded.
  2. Events are matched EXACTLY on the normalized (component, event_name) pair.
     v1 used substring matching with first-match semantics, so some rules were
     never applied (e.g. "informe usuario del curso visto" was captured by
     "curso visto").
  3. Non-student actors are excluded by ROLE, inferred from teacher-only events
     (or supplied as an explicit list), instead of a hard-coded list of names.
  4. Administrative and system-generated events (enrolment, role assignment,
     automatic grading) are flagged in the mapping and excluded, because they
     are not student behavior.
  5. The reference threshold tau_a is computed from the SAME weighted
     frequency freq(u, a) that enters sigma(u, a), over the student population.
  6. Per-area normalized scores (0-100) are reported separately from their
     weighted contributions to the overall score.
  7. Normalization is the within-dataset percentile rank (empirical CDF)
     instead of division by the maximum, which compressed most students
     towards low scores when a few were very active.
  8. tau_a is computed over students with activity in the area (f > 0); the
     share of students with no activity is reported separately.

Reference framework:
  Vuorikari, R., Kluzer, S., & Punie, Y. (2022). DigComp 2.2: The Digital
  Competence Framework for Citizens. https://doi.org/10.2760/115376
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AREAS = {
    "information":   ["1.1", "1.2", "1.3"],
    "communication": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6"],
    "content":       ["3.1", "3.2", "3.3", "3.4"],
    "safety":        ["4.1", "4.2", "4.3", "4.4"],
    "problem":       ["5.1", "5.2", "5.3", "5.4"],
}
SUBCOMPETENCES = [s for subs in AREAS.values() for s in subs]
SUB_TO_AREA = {s: a for a, subs in AREAS.items() for s in subs}

# Events that only a teaching or administrative role can generate in Moodle.
# An actor with at least one of them is treated as non-student.
TEACHER_ONLY_EVENTS = {
    ("sistema", "pregunta borrada"), ("sistema", "pregunta creada"),
    ("sistema", "pregunta actualizada"), ("sistema", "módulo de curso actualizado"),
    ("sistema", "módulo de curso creado"), ("sistema", "calificación borrada"),
    ("cuestionario", "página de edición del cuestionario visitada"),
    ("cuestionario", "informe de cuestionario visto"),
    ("tarea", "formulario de calificaciones visto"),
    ("sistema", "categoría de preguntas visualizada"),
    ("registros", "informe de registro visualizado"),
    ("informe del calificador", "informe del calificador visto"),
    ("tarea", "se ha calificado el envío"),
}

# Behavioral indicators for the textual descriptors D(u).
INDICATORS = {
    "forum_production":  ([("foro", "tema creado")],
                          "Active initiation of academic discussions in the forum"),
    "forum_posting":     ([("foro", "algún contenido ha sido publicado"), ("foro", "mensaje creado")],
                          "Frequent participation in peer discussions"),
    "assignment_submission": ([("tarea", "se ha enviado una entrega"), ("archivos enviados", "entrega creada")],
                          "Consistent submission of academic assignments"),
    "assignment_resubmission": ([("archivos enviados", "envío actualizado"),
                                 ("tarea", "el estado de la entrega se ha actualizado")],
                          "Review and resubmission of assignments after feedback"),
    "feedback_reading":  ([("tarea", "retroalimentación vista")],
                          "Active consultation of teacher feedback"),
    "quiz_review":       ([("cuestionario", "intento del cuestionario revisado")],
                          "Systematic review of assessment results"),
    "quiz_attempt":      ([("cuestionario", "ha comenzado el intento"), ("cuestionario", "intento enviado")],
                          "High participation in digital assessments"),
    "resource_consultation": ([("recurso", "módulo de curso visto"), ("carpeta", "módulo de curso visto"),
                               ("url", "módulo de curso visto"), ("página", "módulo de curso visto")],
                          "Intensive consultation of study materials"),
    "course_access":     ([("sistema", "curso visto")],
                          "High frequency of platform access"),
}


def normalize(text) -> str:
    """Lower-case, collapse whitespace and drop a trailing period."""
    t = re.sub(r"\s+", " ", str(text).strip().lower())
    return t[:-1].strip() if t.endswith(".") else t


@dataclass
class Config:
    """Heuristic configuration (levels 2 and 3 of v1)."""
    area_weights: dict = field(default_factory=lambda: {
        "information": 0.20, "communication": 0.18, "content": 0.17,
        "safety": 0.15, "problem": 0.30})
    multipliers: dict = field(default_factory=lambda: {
        "information": 1.2, "communication": 1.1, "content": 0.8,
        "safety": 0.9, "problem": 1.2})
    tau_quantile: float = 0.75          # tau_a = quantile of freq(u, a) over ACTIVE students (f > 0)
    normalization: str = "percentile"   # "percentile" (within dataset) or "max" (v1)
    robust: float = 0.70
    inferable: float = 0.40
    descriptor_quantile: float = 0.75   # theta_k = quantile of indicator counts
    level_cuts: tuple = (33.0, 66.0)


def load_mapping(path: str) -> pd.DataFrame:
    m = pd.read_csv(path, dtype={"subcompetence": str})
    m["key"] = list(zip(m["component"].map(normalize), m["event_name"].map(normalize)))
    sums = m.groupby("key")["weight"].sum()
    bad = sums[(sums - 1).abs() > 1e-6]
    if len(bad):
        raise ValueError(f"Weights do not sum to 1 for: {list(bad.index)}")
    unknown = set(m["subcompetence"]) - set(SUBCOMPETENCES)
    if unknown:
        raise ValueError(f"Unknown subcompetence codes: {unknown}")
    return m


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: actor, component, event_name (pseudonymized log)."""
    out = df.copy()
    out["key"] = list(zip(out["component"].map(normalize), out["event_name"].map(normalize)))
    return out


def identify_non_students(df: pd.DataFrame, explicit: set | None = None) -> set:
    staff = set(df.loc[df["key"].isin(TEACHER_ONLY_EVENTS), "actor"].dropna())
    if "admin_or_system" in df.columns:
        staff |= set(df.loc[df["admin_or_system"].astype(str) == "True", "actor"].dropna())
    return staff | (explicit or set())


def run(df: pd.DataFrame, mapping: pd.DataFrame, cfg: Config = Config(),
        staff: set | None = None, tau_override: dict | None = None):
    """Return (per-student table, diagnostics dict)."""
    df = prepare(df) if "key" not in df.columns else df
    staff = identify_non_students(df) if staff is None else staff
    st = df[df["actor"].notna() & ~df["actor"].isin(staff)]
    if "origin" in st.columns:   # 'cli' = scheduled tasks (e.g. auto-closing overdue attempts)
        st = st[st["origin"].astype(str).str.lower() != "cli"]
    students = pd.Index(sorted(st["actor"].unique()), name="actor")

    behavioral = mapping[mapping["student_behavior"] == "yes"]
    mapped_keys = set(mapping["key"])
    behavioral_keys = set(behavioral["key"])

    # Stage 1: freq_sub(u, s) = sum of g(e, s) over the student's events
    counts = st.groupby(["actor", "key"]).size().rename("n").reset_index()
    joined = counts.merge(behavioral[["key", "subcompetence", "weight"]], on="key")
    joined["w"] = joined["n"] * joined["weight"]
    freq_sub = (joined.pivot_table(index="actor", columns="subcompetence", values="w",
                                   aggfunc="sum", fill_value=0.0)
                .reindex(index=students, columns=SUBCOMPETENCES, fill_value=0.0))

    # Stage 2: freq(u, a)
    freq = pd.DataFrame({a: freq_sub[subs].sum(axis=1) for a, subs in AREAS.items()})

    # Normalization to [0, 100] by area, within the dataset.
    # "percentile": empirical CDF, norm(u,a) = 100 * share of students with f <= f(u,a);
    #               students with no activity in the area receive 0.
    # "max":        v1 behavior, f / max(f) * 100 (sensitive to extreme values).
    def _norm(c):
        if c.max() <= 0:
            return c * 0.0
        if cfg.normalization == "max":
            return c / c.max() * 100
        return (c.rank(method="max", pct=True) * 100).where(c > 0, 0.0)
    norm = freq.apply(_norm)

    # Reference thresholds tau_a on the same weighted frequency, computed over the
    # students with at least some activity in the area (f > 0). An area in which no
    # student has activity is unobservable in that dataset (tau = 0, sigma = 0).
    zero_share = {a: float((freq[a] <= 0).mean()) for a in AREAS}
    tau = tau_override or {a: (float(freq.loc[freq[a] > 0, a].quantile(cfg.tau_quantile))
                               if (freq[a] > 0).any() else 0.0) for a in AREAS}

    def sigma_col(f, t):
        if t <= 0:          # area unobservable in this dataset
            return pd.Series(0.0, index=f.index)
        return np.minimum(1.0, np.log1p(f) / np.log1p(t))

    sigma = pd.DataFrame({a: sigma_col(freq[a], tau[a]) for a in AREAS})
    obs = sigma.apply(lambda c: np.where(c >= cfg.robust, "robust",
                                np.where(c >= cfg.inferable, "inferable", "scarce")))

    w = pd.Series(cfg.area_weights) * pd.Series(cfg.multipliers)
    score_base = (norm[list(w.index)] * w).sum(axis=1) / w.sum()
    ws = sigma[list(w.index)] * w
    score_sigma = ((norm[list(w.index)] * ws).sum(axis=1) / ws.sum(axis=1)).fillna(0.0)
    lo, hi = cfg.level_cuts
    level = pd.cut(score_sigma, [-np.inf, lo, hi, np.inf], right=False,
                   labels=["Low", "Medium", "High"])

    # Descriptors D(u)
    ind_counts = pd.DataFrame(index=students)
    for k, (events, _) in INDICATORS.items():
        sub = st[st["key"].isin(set(events))]
        ind_counts[k] = sub.groupby("actor").size().reindex(students, fill_value=0)
    # theta_k over students who show the behavior at least once (same rule as tau_a);
    # a descriptor is never activated by a behavior nobody shows.
    theta = {k: (max(1.0, float(ind_counts.loc[ind_counts[k] > 0, k].quantile(cfg.descriptor_quantile)))
                 if (ind_counts[k] > 0).any() else np.inf) for k in INDICATORS}
    def describe(row):
        act = [INDICATORS[k][1] for k in INDICATORS if row[k] >= theta[k]]
        return "; ".join(act) if act else "Insufficient activity to generate behavioral descriptors"
    descriptors = ind_counts.apply(describe, axis=1)

    out = pd.concat([
        norm.add_prefix("norm_"), freq.add_prefix("freq_"),
        sigma.add_prefix("sigma_"), obs.add_prefix("obs_"),
        freq_sub.add_prefix("sub_"),
        pd.DataFrame({"score_base": score_base, "score_sigma": score_sigma,
                      "level": level, "descriptors": descriptors}),
    ], axis=1)

    ev_keys = st["key"]
    diag = {
        "n_events_total": int(len(df)),
        "n_events_students": int(len(st)),
        "n_students": int(len(students)),
        "n_non_student_actors": int(len(staff)),
        "share_student_events_in_mapping": float(ev_keys.isin(mapped_keys).mean()),
        "share_student_events_behavioral": float(ev_keys.isin(behavioral_keys).mean()),
        "distinct_pairs_students": int(ev_keys.nunique()),
        "unmapped_pairs": st.loc[~ev_keys.isin(mapped_keys), "key"].value_counts(),
        "subcompetences_without_signal": [s for s in SUBCOMPETENCES if freq_sub[s].sum() == 0],
        "tau": tau, "theta": theta, "zero_share": zero_share,
    }
    return out, diag


def observability_table(out: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({a: out[f"obs_{a}"].value_counts(normalize=True)
                         .reindex(["robust", "inferable", "scarce"], fill_value=0) * 100
                         for a in AREAS}).T.round(1)
