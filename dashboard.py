"""
RetailVision AI — Streamlit Dashboard
Run with: streamlit run dashboard.py
"""

import os
from datetime import datetime, timedelta

import requests
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RetailVision AI Dashboard", layout="wide")

st.title("🛍️ RetailVision AI — Store Analytics Dashboard")

with st.sidebar:
    st.header("Settings")
    store_id = st.text_input("Store ID", value="store_1")
    since_hours = st.slider("Look back (hours)", min_value=1, max_value=168, value=24)
    refresh = st.button("🔄 Refresh now")
    auto_refresh = st.checkbox("Auto-refresh every 10s", value=False)


def fetch_json(path, params=None):
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


summary, err = fetch_json(f"/footfall/summary/{store_id}", {"since_hours": since_hours})

if err:
    st.error(f"Could not reach the API at {API_BASE}. Is `uvicorn api:app --port 8000` running?\n\nDetails: {err}")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entries", summary["entries"])
    col2.metric("Total Exits", summary["exits"])
    col3.metric("Current Occupancy", summary["current_occupancy"])

    st.divider()
    st.subheader("📈 Hourly Footfall")

    hourly, err2 = fetch_json(f"/footfall/hourly/{store_id}", {"since_hours": since_hours})
    if err2:
        st.warning(f"Could not load hourly data: {err2}")
    elif hourly and hourly.get("data"):
        df = pd.DataFrame(hourly["data"])
        df["hour"] = pd.to_datetime(df["hour"])
        df = df.set_index("hour")[["entries", "exits"]]
        st.bar_chart(df)

        st.subheader("Raw Data")
        st.dataframe(df.reset_index(), use_container_width=True)
    else:
        st.info("No footfall data yet for this period. Run the counter script to generate some.")

st.divider()
st.subheader("📦 Shelf Inventory Status")

shelf, err3 = fetch_json(f"/shelf/latest/{store_id}")
if err3:
    st.warning(f"Could not load shelf data: {err3}")
elif shelf and shelf.get("zones"):
    zdf = pd.DataFrame(shelf["zones"])
    for _, row in zdf.iterrows():
        icon = "🔴" if row["status"] == "low_stock" else "🟢"
        st.write(f"{icon} **{row['zone_name']}** — {int(row['fill_ratio']*100)}% full ({row['status']})")
    st.dataframe(zdf, use_container_width=True)
else:
    st.info("No shelf data yet. Run `shelf_monitor.py` to generate some.")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · API: {API_BASE}")


if auto_refresh:
    import time
    time.sleep(10)
    st.rerun()
