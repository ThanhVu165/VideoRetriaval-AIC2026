from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aic2026.beit3 import BEiT3Index


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and smoke-test the aligned BEiT-3 artifact")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", type=str)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    index = BEiT3Index.from_files(args.manifest, args.index)
    print(json.dumps(index.diagnostics(), indent=2))

    if args.query:
        try:
            from aic2026.beit3_runtime import BEiT3QueryEncoder
        except ImportError as exc:
            raise SystemExit(
                "BEiT-3 artifact audit passed, but query encoding is not installed. "
                "Use the artifact diagnostics only, or install the runtime/model package."
            ) from exc

        encoder = BEiT3QueryEncoder()
        query_embedding = encoder.encode_one(args.query)
        rows = index.search(query_embedding, top_k=args.top_k)
        print(rows.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
