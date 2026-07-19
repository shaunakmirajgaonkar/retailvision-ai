"""
RetailVision AI — Shelf Inventory & Product Detection Module
-----------------------------------------------------------------
Monitors shelf images/video for product presence, detects empty/low-stock
shelf zones, and logs inventory snapshots to PostgreSQL.

Approach:
    Since a generic COCO-pretrained YOLO model doesn't know specific product
    SKUs, this module uses a two-tier strategy:
      1. If you have a custom-trained product-detection model (recommended
         for production — train on your own shelf photos with tools like
         Roboflow or Ultralytics HUB), point SHELF_MODEL_PATH at it and it
         will detect and count actual product classes.
      2. Out of the box (no custom model yet), it falls back to a generic
         "shelf gap" detector: divides each shelf zone into a grid, and uses
         classical CV (edge density + contour analysis) to flag empty vs
         stocked cells. This gives you a working empty-shelf alert system on
         day one, which you can upgrade once you have labeled data.

Dependencies:
    pip install ultralytics opencv-python numpy psycopg2-binary
"""

import os
import json
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SHELF_MODEL_PATH = os.getenv("SHELF_MODEL_PATH", "")  # empty = use fallback gap detector
CONF_THRESHOLD = float(os.getenv("SHELF_CONF_THRESHOLD", 0.35))

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "retailvision"),
    "user": os.getenv("DB_USER", os.getenv("USER", "postgres")),
    "password": os.getenv("DB_PASSWORD", ""),
}


# --------------------------------------------------------------------------
# Shelf zone definition
# --------------------------------------------------------------------------
class ShelfZone:
    """A rectangular region of interest on a shelf (e.g. one shelf row or bay)."""

    def __init__(self, name, x1, y1, x2, y2, min_stock_ratio=0.3):
        self.name = name
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.min_stock_ratio = min_stock_ratio  # below this fill ratio = "low stock"

    def crop(self, frame):
        return frame[self.y1:self.y2, self.x1:self.x2]


# --------------------------------------------------------------------------
# Fallback classical-CV empty-shelf detector (no training data needed)
# --------------------------------------------------------------------------
def estimate_fill_ratio(zone_img, grid=(4, 3)):
    """
    Splits the zone into a grid and estimates how many cells look 'stocked'
    (high edge density / texture = products present) vs 'empty' (flat,
    uniform region = bare shelf/pegboard).
    Returns a fill ratio between 0 (empty) and 1 (fully stocked).
    """
    if zone_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows, cols = grid
    cell_h, cell_w = h // rows, w // cols

    stocked_cells = 0
    total_cells = rows * cols

    for r in range(rows):
        for c in range(cols):
            cell = gray[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            if cell.size == 0:
                continue
            edges = cv2.Canny(cell, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            variance = cell.var()
            # heuristic: products create edges + variance; bare shelf is smooth
            if edge_density > 0.05 or variance > 200:
                stocked_cells += 1

    return stocked_cells / total_cells if total_cells else 0.0


# --------------------------------------------------------------------------
# Main shelf monitor
# --------------------------------------------------------------------------
class ShelfMonitor:
    def __init__(self, zones, store_id="store_1", model_path=None):
        """
        zones: list of ShelfZone objects
        model_path: path to a custom-trained product detection model (optional)
        """
        self.zones = zones
        self.store_id = store_id
        self.model = None

        model_path = model_path or SHELF_MODEL_PATH
        if model_path and YOLO is not None:
            self.model = YOLO(model_path)

    def analyze_frame(self, frame):
        """Returns a list of dicts: one status per zone."""
        results = []
        for zone in self.zones:
            crop = zone.crop(frame)

            if self.model is not None:
                detections = self.model(crop, conf=CONF_THRESHOLD, verbose=False)[0]
                product_count = len(detections.boxes)
                # crude fill estimate from detection density relative to zone area
                fill_ratio = min(1.0, product_count / 10.0)
                method = "custom_model"
            else:
                fill_ratio = estimate_fill_ratio(crop)
                product_count = None
                method = "fallback_cv"

            status = "low_stock" if fill_ratio < zone.min_stock_ratio else "stocked"
            results.append({
                "zone": zone.name,
                "fill_ratio": round(fill_ratio, 2),
                "product_count": product_count,
                "status": status,
                "method": method,
            })
        return results

    def annotate(self, frame, results):
        for zone, res in zip(self.zones, results):
            color = (0, 0, 255) if res["status"] == "low_stock" else (0, 200, 0)
            cv2.rectangle(frame, (zone.x1, zone.y1), (zone.x2, zone.y2), color, 2)
            label = f"{zone.name}: {res['status']} ({int(res['fill_ratio']*100)}%)"
            cv2.putText(frame, label, (zone.x1, max(20, zone.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def log_snapshot(self, results):
        if psycopg2 is None:
            return
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            for res in results:
                cur.execute(
                    """
                    INSERT INTO shelf_snapshots
                        (store_id, zone_name, fill_ratio, product_count, status, method, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        self.store_id, res["zone"], res["fill_ratio"],
                        res["product_count"], res["status"], res["method"],
                        datetime.now(),
                    ),
                )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[DB WARNING] Could not log shelf snapshot: {e}")

    def run_on_image(self, image_path, save_annotated=None):
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        results = self.analyze_frame(frame)
        self.log_snapshot(results)
        annotated = self.annotate(frame.copy(), results)

        if save_annotated:
            cv2.imwrite(save_annotated, annotated)

        return results, annotated

    def run_live(self, source=0, interval_seconds=30, display=True):
        """Continuously monitor a camera feed, checking shelves every N seconds."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        last_check = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                now = datetime.now().timestamp()
                if now - last_check >= interval_seconds:
                    results = self.analyze_frame(frame)
                    self.log_snapshot(results)
                    for res in results:
                        if res["status"] == "low_stock":
                            print(f"⚠️  ALERT: {res['zone']} is low stock ({int(res['fill_ratio']*100)}% full)")
                    last_check = now
                    annotated = self.annotate(frame.copy(), results)
                else:
                    annotated = frame

                if display:
                    cv2.imshow("RetailVision AI — Shelf Monitor", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RetailVision AI Shelf Monitor")
    parser.add_argument("--zones-config", help="Path to JSON file defining shelf zones", default=None)
    parser.add_argument("--image", help="Path to a single shelf image to analyze")
    parser.add_argument("--source", default=0, help="Camera index or video source for live monitoring")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between checks in live mode")
    parser.add_argument("--store-id", default="store_1")
    parser.add_argument("--save", default=None, help="Path to save annotated output image")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    # Default zones if no config file given: three horizontal shelf rows
    if args.zones_config:
        with open(args.zones_config) as f:
            zone_defs = json.load(f)
        zones = [ShelfZone(**z) for z in zone_defs]
    else:
        zones = [
            ShelfZone("top_shelf", 0, 0, 640, 160),
            ShelfZone("middle_shelf", 0, 160, 640, 320),
            ShelfZone("bottom_shelf", 0, 320, 640, 480),
        ]

    monitor = ShelfMonitor(zones=zones, store_id=args.store_id)

    if args.image:
        results, annotated = monitor.run_on_image(args.image, save_annotated=args.save)
        print(json.dumps(results, indent=2))
    else:
        source = args.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        monitor.run_live(source=source, interval_seconds=args.interval, display=not args.no_display)
