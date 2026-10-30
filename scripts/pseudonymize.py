"""
pseudonymize.py
===============
Converts a Moodle log export (Spanish-language interface, CSV) into the pseudonymized format
used by the pipeline. User identifiers are replaced by salted SHA-256 hashes, and names, IP
addresses and free-text descriptions are dropped.

The salt is secret: choose your own value and keep it outside the repository, otherwise the
pseudonyms of a published dataset could be linked back to Moodle user ids.

Usage:
    DCA_SALT="your-secret-value" python pseudonymize.py moodle_export.csv pseudonymized.csv

Input columns (Spanish Moodle export): Hora, Nombre completo del usuario, Contexto del evento,
Componente, Nombre evento, Descripción, Origen.
Output columns: timestamp, actor, related_user, context, component, event_name, origin,
admin_or_system.
"""
import os, sys, re, hashlib
import pandas as pd

SALT = os.environ.get("DCA_SALT")
if not SALT:
    sys.exit("Set a secret salt in the DCA_SALT environment variable.")


def uid(desc):
    m = re.search(r"The user with id '(-?\d+)'", str(desc))
    return m.group(1) if m else None


def related(desc):
    ids = re.findall(r"user with id '(-?\d+)'", str(desc))
    return ids[1] if len(ids) > 1 else None


def h(x):
    return None if (x is None or not isinstance(x, str)) else "u_" + hashlib.sha256((SALT + x).encode()).hexdigest()[:10]


def main(src, dst):
    df = pd.read_csv(src, encoding="utf-8-sig", dtype=str)
    actor = df["Descripción"].map(uid).map(h)
    rel = df["Descripción"].map(related).map(h)
    # accounts named as administrators or system ("-") in the export
    admin = df["Nombre completo del usuario"].str.lower().isin(["-", "administrador", "unne administrador"])
    out = pd.DataFrame({
        "timestamp": df["Hora"], "actor": actor, "related_user": rel,
        "context": df["Contexto del evento"].str.replace(r"\s+", " ", regex=True),
        "component": df["Componente"], "event_name": df["Nombre evento"],
        "origin": df["Origen"], "admin_or_system": admin,
    })
    out.to_csv(dst, index=False)
    print(f"rows: {len(out)} | without actor id: {out['actor'].isna().sum()} | distinct actors: {out['actor'].nunique()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
