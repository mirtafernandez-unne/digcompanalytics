# DIGCOMPAnalytics

Traceable, observability-aware inference of DigComp 2.2 behavioral proxies from Moodle activity logs.

This repository contains the implementation and the evaluation code of the article:

> M. G. Fernández, S. I. Mariño, M. V. Godoy Guglielmone, and W. G. Barrios, "A traceable framework for
> observability-aware inference of DigComp 2.2 behavioral proxies from Moodle logs," submitted to *IEEE Access*.

## What the framework does

DIGCOMPAnalytics turns a Moodle log export into behavioral proxy indicators organized by the
21 subcompetences of DigComp 2.2, and reports for each student and area how much behavioral
evidence supports the inference (observability index σ). The pipeline has five stages:

1. **Actor filtering**: teachers are identified through 13 teacher-only actions, administrative
   accounts and scheduled-task events (`origin = cli`) are excluded, and non-behavioral records
   (enrolment, role assignment, automatic grading, quiz access-rule denials) are kept for
   traceability but excluded from inference.
2. **Weighted event mapping**: each (component, event) pair is mapped to a probability
   distribution over subcompetences (`config/mapping_digcomp22.csv`, editable).
3. **Two-level aggregation and within-cohort percentile normalization.**
4. **Observability index** σ = min(1, ln(1+f)/ln(1+τ)), with τ the 75th percentile among active students;
   Robust ≥ 0.70, Inferable 0.40–0.70, Scarce < 0.40.
5. **Composite score, relative activity level (low / medium / high) and textual descriptors.**

## Repository structure

| Path | Content |
| --- | --- |
| `src/digcompanalytics.py` | The framework (pipeline, configuration, observability) |
| `src/analysis.py` | Evaluation: corpus, observability under three configurations, K-means (k = 2..7), baselines B1 and B2, bootstrap stability, PCA, sequential patterns, cross-cohort recurrence |
| `src/transfer_operational.py` | Operational transfer test (target expressed through the source distribution) |
| `src/robustness.py` | Robustness analyses: k = 3 stability, area selection, mapping-weight perturbation (Dirichlet), discriminant evidence |
| `config/mapping_digcomp22.csv` | Event mapping: 151 Moodle (component, event) pairs, Spanish-language interface |
| `scripts/pseudonymize.py` | Converts a raw Moodle export into the pseudonymized input format |
| `scripts/make_synthetic.py` | Generates synthetic logs from aggregate profiles (`synthetic_profiles.json`) |
| `data/synthetic/` | Synthetic LED (three cohorts) and ITAA logs |
| `run_all.sh` | Runs the complete evaluation |

## Data availability

The original logs contain personal data of students and cannot be shared. The repository
includes **synthetic logs** with the same structure (columns, event vocabulary, observation
windows, number of students, distribution of activity and mix of event types per course),
generated from aggregate figures only. They allow the complete pipeline to run, but they do
**not** reproduce the numerical results of the article.

## Quick start

```bash
pip install -r requirements.txt
./run_all.sh                                   # synthetic data, full evaluation
DCA_N_PERT=20 DCA_N_BOOT=20 ./run_all.sh       # quick run (fewer perturbations and resamples)
```

To analyze your own course, export the Moodle log (CSV, Spanish interface), pseudonymize it and run:

```bash
DCA_SALT="a-secret-value" python scripts/pseudonymize.py export.csv data/real/course.csv
./run_all.sh data/real/led.csv data/real/itaa.csv
```

The LED file may contain several academic years: each student is assigned to the cohort of the
year of their first event, and only that year's events are used. Results are written to `results/`.

## Reproducibility notes

- Python 3.13; package versions in `requirements.txt`. Random seed 2026 throughout.
- Each row of the export is one event; rows without an identifiable user are discarded.
- Percentile ties receive the highest rank of their group; students without activity in an area receive 0.
- K-means uses 20 initializations; bootstrap stability uses 200 resamples; weight perturbation uses 500 runs per level.
- Baseline B2 assigns each event to its highest-weight subcompetence; ties are broken by the lowest subcompetence code.

## License

Code released under the MIT License (see `LICENSE`). The mapping configuration is released under CC BY 4.0.

## Citation

See `CITATION.cff`.
