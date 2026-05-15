import asyncio
import csv
import io
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env file from script directory if present (no extra deps needed)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

CSV_URL = "https://raw.githubusercontent.com/rongyinger/api-checker/main/docs/data.csv"
MODELS_URL = "https://aiplatform.zjsk.cc/api/pricing"
LOCAL_CSV = Path(__file__).parent / "docs" / "data.csv"
REFRESH_INTERVAL = 300  # seconds

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict = {"rows": [], "updated_at": None}


def _parse_csv(text: str) -> list[dict]:
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            latency = float(row.get("latency", -1))
        except (ValueError, TypeError):
            latency = -1.0
        rows.append({
            "timestamp": row.get("timestamp", ""),
            "model": row.get("model", ""),
            "latency": latency,
            "status": row.get("status", ""),
            "error": row.get("error", ""),
        })
    return rows


async def _fetch_live_models(client: httpx.AsyncClient) -> list[str]:
    """Fetch current model list from the platform API. Returns empty list on failure."""
    try:
        resp = await client.get(MODELS_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        models = [m["model_name"] for m in data if m.get("model_name")]
        if models:
            print(f"[INFO] Live model list: {len(models)} models")
        return models
    except Exception as e:
        print(f"[WARN] Failed to fetch live model list: {e}")
        return []


def _merge_with_live_models(rows: list[dict], live_models: list[str]) -> list[dict]:
    """Add placeholder rows for live models that have no CSV history."""
    if not live_models:
        return rows

    models_in_csv = {r["model"] for r in rows}
    # Latest timestamp in CSV (used as placeholder timestamp)
    latest_ts = max((r["timestamp"] for r in rows), default="") if rows else ""

    new_rows = list(rows)
    added = 0
    for model in live_models:
        if model not in models_in_csv:
            new_rows.append({
                "timestamp": latest_ts,
                "model": model,
                "latency": -1.0,
                "status": "unknown",
                "error": "no data yet",
            })
            added += 1

    if added:
        print(f"[INFO] Added {added} placeholder rows for new models")
    return new_rows


async def fetch_and_cache() -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Fetch CSV (try GitHub first, fall back to local)
        csv_rows: list[dict] = []
        try:
            resp = await client.get(CSV_URL)
            resp.raise_for_status()
            csv_rows = _parse_csv(resp.text)
            print(f"[INFO] Fetched {len(csv_rows)} rows from GitHub")
        except Exception as e:
            print(f"[WARN] GitHub CSV fetch failed: {e}. Trying local file.")
            if LOCAL_CSV.exists():
                try:
                    csv_rows = _parse_csv(LOCAL_CSV.read_text(encoding="utf-8"))
                    print(f"[INFO] Loaded {len(csv_rows)} rows from local file")
                except Exception as e2:
                    print(f"[ERROR] Local file read failed: {e2}")

        # 2. Fetch live model list
        live_models = await _fetch_live_models(client)

        # 3. Merge
        rows = _merge_with_live_models(csv_rows, live_models)

        _cache["rows"] = rows
        _cache["updated_at"] = datetime.now(timezone.utc).isoformat()
        print(f"[INFO] Cache updated: {len(rows)} total rows at {_cache['updated_at']}")


async def refresh_loop() -> None:
    while True:
        await fetch_and_cache()
        await asyncio.sleep(REFRESH_INTERVAL)


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(refresh_loop())


@app.get("/api/status")
async def get_status() -> dict:
    return {
        "updated_at": _cache["updated_at"],
        "rows": _cache["rows"],
    }




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
