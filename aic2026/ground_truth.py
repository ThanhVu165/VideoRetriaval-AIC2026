from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_query_ids(path: str | Path, query_column: str = "Description") -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(payload if isinstance(payload, list) else payload["queries"])
    else:
        raise ValueError(f"Unsupported query file format: {path.suffix}")

    id_candidates = ["Query Name", "query_id", "id", "name"]
    qid_col = next((c for c in id_candidates if c in df.columns), None)
    if qid_col is None:
        raise ValueError(f"No query ID column found. Columns: {df.columns.tolist()}")
    if query_column not in df.columns:
        raise ValueError(f"Query column {query_column!r} not found. Columns: {df.columns.tolist()}")

    out = pd.DataFrame({
        "query_id": df[qid_col].astype(str),
        "query": df[query_column].fillna("").astype(str),
    })
    out = out[out["query"].str.strip().ne("")].drop_duplicates("query_id").reset_index(drop=True)
    return out


def infer_task_type(query_id: str) -> str:
    q = query_id.lower()
    if q.startswith("trake"):
        return "TRAKE"
    if q.startswith("qa"):
        return "QA"
    if q.startswith("tkis"):
        return "TKIS"
    return "UNKNOWN"


def build_ground_truth_template(
    queries_path: str | Path,
    output_path: str | Path,
    query_column: str = "Description",
) -> dict[str, int]:
    queries = load_query_ids(queries_path, query_column=query_column)
    records = []
    for _, row in queries.iterrows():
        qid = str(row["query_id"])
        records.append(
            {
                "query_id": qid,
                "task_type": infer_task_type(qid),
                "video_ids": [],
                "frames": {},
                "intervals": {},
                "events": [],
                "notes": "FILL WITH OFFICIAL/MANUALLY VERIFIED GROUND TRUTH. Do not copy predictions here.",
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"queries": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = pd.Series([r["task_type"] for r in records]).value_counts().to_dict() if records else {}
    return {"queries": len(records), **{str(k).lower(): int(v) for k, v in counts.items()}}
