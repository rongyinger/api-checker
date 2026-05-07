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

CSV_URL = "https://raw.githubusercontent.com/rongyinger/api-checker/main/docs/data.csv"
# Local CSV path (relative to this script's directory)
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


async def fetch_and_cache() -> None:
    # 1. Try GitHub raw
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(CSV_URL)
            resp.raise_for_status()
            rows = _parse_csv(resp.text)
            _cache["rows"] = rows
            _cache["updated_at"] = datetime.now(timezone.utc).isoformat()
            print(f"[INFO] Fetched {len(rows)} rows from GitHub at {_cache['updated_at']}")
            return
    except Exception as e:
        print(f"[WARN] GitHub fetch failed: {e}. Falling back to local file.")

    # 2. Fallback: read local CSV
    if LOCAL_CSV.exists():
        try:
            rows = _parse_csv(LOCAL_CSV.read_text(encoding="utf-8"))
            _cache["rows"] = rows
            _cache["updated_at"] = datetime.now(timezone.utc).isoformat()
            print(f"[INFO] Loaded {len(rows)} rows from local file at {_cache['updated_at']}")
        except Exception as e:
            print(f"[ERROR] Local file read failed: {e}")
    else:
        print(f"[WARN] Local file not found: {LOCAL_CSV}")


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
