"""
RetailVision AI — FastAPI service for footfall analytics.
Run with: uvicorn api:app --reload --port 8000
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import psycopg2
import psycopg2.extras

from footfall_counter import DB_CONFIG

app = FastAPI(title="RetailVision AI - Footfall API", version="1.0")


class FootfallSummary(BaseModel):
    store_id: str
    entries: int
    exits: int
    current_occupancy: int


def get_conn():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/footfall/summary/{store_id}", response_model=FootfallSummary)
def footfall_summary(store_id: str, since_hours: int = Query(24, ge=1, le=720)):
    since = datetime.now() - timedelta(hours=since_hours)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE event_type = 'entry') AS entries,
                COUNT(*) FILTER (WHERE event_type = 'exit') AS exits
            FROM footfall_events
            WHERE store_id = %s AND timestamp >= %s
            """,
            (store_id, since),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    entries = row["entries"] or 0
    exits = row["exits"] or 0
    return FootfallSummary(
        store_id=store_id,
        entries=entries,
        exits=exits,
        current_occupancy=max(0, entries - exits),
    )


@app.get("/footfall/hourly/{store_id}")
def footfall_hourly(store_id: str, since_hours: int = Query(24, ge=1, le=720)):
    since = datetime.now() - timedelta(hours=since_hours)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT hour, entries, exits
            FROM footfall_hourly
            WHERE store_id = %s AND hour >= %s
            ORDER BY hour
            """,
            (store_id, since),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {"store_id": store_id, "data": rows}


@app.get("/shelf/latest/{store_id}")
def shelf_latest(store_id: str):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT ON (zone_name)
                zone_name, fill_ratio, product_count, status, method, timestamp
            FROM shelf_snapshots
            WHERE store_id = %s
            ORDER BY zone_name, timestamp DESC
            """,
            (store_id,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {"store_id": store_id, "zones": rows}


@app.get("/shelf/history/{store_id}/{zone_name}")
def shelf_history(store_id: str, zone_name: str, since_hours: int = Query(24, ge=1, le=720)):
    since = datetime.now() - timedelta(hours=since_hours)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT fill_ratio, product_count, status, timestamp
            FROM shelf_snapshots
            WHERE store_id = %s AND zone_name = %s AND timestamp >= %s
            ORDER BY timestamp
            """,
            (store_id, zone_name, since),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {"store_id": store_id, "zone": zone_name, "data": rows}

