# RetailVision AI — Customer Counting, Footfall & Shelf Inventory

100% local, no cloud dependency. Runs on CPU or GPU.

## Setup

```bash
pip3 install ultralytics opencv-python numpy psycopg2-binary fastapi uvicorn pydantic streamlit pandas requests --break-system-packages
```

1. Create the database and tables:
```bash
createdb retailvision
psql -d retailvision -f schema.sql
```

2. Set environment variables (adjust for your local Postgres setup — on macOS with Homebrew, DB_USER is usually your Mac username and DB_PASSWORD is empty):
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=retailvision
export DB_USER=$(whoami)
export DB_PASSWORD=""
export YOLO_MODEL_PATH=yolov8n.pt
```

## 1. Footfall Counter (people entering/exiting)

```bash
python3 footfall_counter.py --source 0 --orientation horizontal --line-position 0.5 --store-id store_1
```

Use a video file or RTSP stream instead of a webcam:
```bash
python3 footfall_counter.py --source rtsp://camera-ip/stream --store-id store_1 --save output.mp4
```

Press `q` to quit the live window.

## 2. Shelf Inventory Monitor (empty-shelf / low-stock detection)

Analyze a single shelf photo:
```bash
python3 shelf_monitor.py --image shelf_photo.jpg --store-id store_1 --save annotated.jpg
```

Monitor a live camera feed, checking every 30 seconds:
```bash
python3 shelf_monitor.py --source 0 --interval 30 --store-id store_1
```

By default it splits the frame into 3 horizontal zones (top/middle/bottom shelf) and uses a classical CV
"fill ratio" heuristic (edge density + variance) to flag low stock — no training data needed to get started.

To use your own shelf zones, create a JSON file:
```json
[
  {"name": "cereal_aisle_top", "x1": 0, "y1": 0, "x2": 400, "y2": 200, "min_stock_ratio": 0.3},
  {"name": "cereal_aisle_bottom", "x1": 0, "y1": 200, "x2": 400, "y2": 400, "min_stock_ratio": 0.3}
]
```
and run with `--zones-config zones.json`.

To upgrade to real product detection later, train a custom YOLO model on your own shelf/product
photos (e.g. via Roboflow or Ultralytics HUB) and point `SHELF_MODEL_PATH` at the trained weights —
the module will automatically switch from the fallback heuristic to real product counts.

## 3. API (serves both footfall and shelf data)

```bash
uvicorn api:app --reload --port 8000
```

- `GET /footfall/summary/{store_id}?since_hours=24`
- `GET /footfall/hourly/{store_id}?since_hours=24`
- `GET /shelf/latest/{store_id}`
- `GET /shelf/history/{store_id}/{zone_name}?since_hours=24`

## 4. Dashboard

```bash
streamlit run dashboard.py
```

Opens in your browser at `http://localhost:8501`. Shows live entries/exits/occupancy, an hourly
footfall chart, and current shelf status per zone. Requires the API (`uvicorn api:app`) to be running.

## Notes

- Uses YOLOv8n (nano) for speed on CPU; swap `YOLO_MODEL_PATH` for a larger model (yolov8s/m) if you have a GPU and want higher accuracy.
- Footfall tracking is a lightweight centroid tracker (no external dependency) — good enough for line-crossing counts.
- Shelf monitoring works out of the box with classical CV; upgrade to a custom-trained model for real product-level counts.
