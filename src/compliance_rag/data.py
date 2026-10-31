"""Clause corpus and labeled benchmark loading."""
from pathlib import Path
from typing import List
import pandas as pd
from .schemas import Clause, Verdict

def load_clauses(path: str) -> List[Clause]:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"clause", "swe_id", "requirement"}
    if not required <= set(df.columns) or df.empty:
        raise ValueError(f"Clause corpus must contain rows and columns {sorted(required)}: {path}")
    return [Clause(clause_id=r.clause.strip(), swe_id=r.swe_id.strip(), text=r.requirement.strip())
            for r in df.itertuples()]


def load_benchmark(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"requirement_id", "requirement_text", "clause_id", "label"}
    if not required <= set(df.columns) or df.empty:
        raise ValueError(f"Benchmark must contain rows and columns {sorted(required)}: {path}")
    df["clause_id"] = df["clause_id"].str.strip()
    df["label"] = df["label"].str.strip().str.capitalize()
    bad = set(df["label"]) - {v.value for v in Verdict}
    if bad:
        raise ValueError(f"Unexpected labels in benchmark: {bad}")
    return df


def discover_csvs(directory):
    """Notebook discovery: newest clause corpus and all compatible benchmark CSVs."""
    clauses, benchmarks = [], []
    for path in sorted(Path(directory).glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if {"clause", "swe_id", "requirement"} <= columns:
            clauses.append(path)
        elif {"requirement_text", "clause_id", "label"} <= columns:
            benchmarks.append(path)
    return max(clauses, key=lambda p: p.stat().st_mtime) if clauses else None, benchmarks


def combine_benchmarks(paths):
    """Combine notebook inputs, drop repeated text/clause pairs, and assign fresh IDs."""
    combined = pd.concat([pd.read_csv(p, dtype=str).fillna("") for p in paths], ignore_index=True)
    combined = combined.drop_duplicates(subset=["requirement_text", "clause_id"]).reset_index(drop=True)
    combined["requirement_id"] = range(1, len(combined) + 1)
    return combined
