# Architecture

RetailVision AI is composed of four independent, loosely-coupled components:

```
[Camera/Video] --> footfall_counter.py --> PostgreSQL --> api.py --> dashboard.py
                --> shelf_monitor.py    -->      |
```

## Components

### 1. `footfall_counter.py`
- Reads frames from a webcam, RTSP stream, or video file.
- Runs YOLOv8 (person class only) for detection.
- Tracks detections frame-to-frame with a lightweight centroid tracker.
- Detects line crossings to count entries/exits and maintain live occupancy.
- Logs each entry/exit event to the `footfall_events` table.

### 2. `shelf_monitor.py`
- Divides a shelf image/frame into configurable zones.
- Falls back to a classical CV heuristic (edge density + variance) to estimate
  fill ratio when no custom-trained model is available.
- Optionally uses a custom YOLO model (`SHELF_MODEL_PATH`) for real product counts.
- Logs snapshots to the `shelf_snapshots` table.

### 3. `api.py`
- FastAPI service exposing read endpoints over the PostgreSQL data:
  footfall summary/hourly, shelf latest/history.
- Decouples the dashboard (or any other client) from direct DB access.

### 4. `dashboard.py`
- Streamlit app that polls the API and renders live metrics, charts, and
  shelf status.

## Data Flow

1. Detection scripts run continuously (or on-demand) and write events/snapshots
   directly to PostgreSQL.
2. The API reads from PostgreSQL and serves aggregated views.
3. The dashboard polls the API on a refresh interval and renders the UI.

This separation means each piece can be swapped independently — e.g. replacing
the centroid tracker with ByteTrack, or the classical CV shelf detector with a
custom-trained model, without touching the API or dashboard.
