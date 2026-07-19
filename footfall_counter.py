"""
RetailVision AI — Customer Counting & Footfall Analytics Module
-----------------------------------------------------------------
Detects and tracks people in a retail store video/camera feed using YOLOv8,
counts entries/exits across a virtual line, and logs footfall data (hourly
counts, dwell time) to PostgreSQL for the analytics dashboard.

Dependencies:
    pip install ultralytics opencv-python numpy psycopg2-binary python-dotenv

Model:
    Uses YOLOv8n (nano) pretrained on COCO — class 0 = 'person'.
    Swap MODEL_PATH for a custom-trained retail model if you have one.
"""

import os
import time
import math
from datetime import datetime
from collections import OrderedDict

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None  # DB logging becomes a no-op if not installed


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.4))
PERSON_CLASS_ID = 0

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "retailvision"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}


# --------------------------------------------------------------------------
# Simple centroid tracker (lightweight alternative to DeepSORT for local use)
# --------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict()        # object_id -> centroid
        self.disappeared = OrderedDict()    # object_id -> frames missing
        self.first_seen = OrderedDict()     # object_id -> timestamp
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.first_seen[self.next_object_id] = time.time()
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.first_seen[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = math.dist(oc, ic)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# --------------------------------------------------------------------------
# Footfall counter with line-crossing entry/exit logic
# --------------------------------------------------------------------------
class FootfallCounter:
    def __init__(self, source=0, line_position=0.5, orientation="horizontal", store_id="store_1"):
        """
        source: camera index, RTSP URL, or video file path
        line_position: fraction (0-1) of frame height/width where the counting line sits
        orientation: 'horizontal' (line across width, counts vertical movement)
                     or 'vertical' (line across height, counts horizontal movement)
        """
        self.model = YOLO(MODEL_PATH)
        self.source = source
        self.line_position = line_position
        self.orientation = orientation
        self.store_id = store_id

        self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)
        self.track_history = {}   # object_id -> list of centroids
        self.counted_ids = set()

        self.entries = 0
        self.exits = 0
        self.current_occupancy = 0

    def _get_line_coord(self, frame_shape):
        h, w = frame_shape[:2]
        if self.orientation == "horizontal":
            return int(h * self.line_position), w  # y-coordinate of line
        else:
            return int(w * self.line_position), h  # x-coordinate of line

    def _check_crossing(self, object_id, centroid, line_coord):
        history = self.track_history.setdefault(object_id, [])
        history.append(centroid)
        if len(history) > 30:
            history.pop(0)

        if len(history) < 2 or object_id in self.counted_ids:
            return None

        prev = history[-2]
        curr = history[-1]

        if self.orientation == "horizontal":
            prev_val, curr_val = prev[1], curr[1]
        else:
            prev_val, curr_val = prev[0], curr[0]

        if prev_val < line_coord <= curr_val:
            self.counted_ids.add(object_id)
            return "entry"
        elif prev_val > line_coord >= curr_val:
            self.counted_ids.add(object_id)
            return "exit"
        return None

    def process_frame(self, frame):
        results = self.model(frame, conf=CONF_THRESHOLD, classes=[PERSON_CLASS_ID], verbose=False)[0]

        rects = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            rects.append((x1, y1, x2, y2))

        objects = self.tracker.update(rects)
        line_coord, _ = self._get_line_coord(frame.shape)

        for object_id, centroid in objects.items():
            event = self._check_crossing(object_id, tuple(centroid), line_coord)
            if event == "entry":
                self.entries += 1
                self.current_occupancy += 1
                self._log_event("entry")
            elif event == "exit":
                self.exits += 1
                self.current_occupancy = max(0, self.current_occupancy - 1)
                self._log_event("exit")

        annotated = self._annotate(frame, rects, objects, line_coord)
        return annotated

    def _annotate(self, frame, rects, objects, line_coord):
        h, w = frame.shape[:2]
        if self.orientation == "horizontal":
            cv2.line(frame, (0, line_coord), (w, line_coord), (0, 255, 255), 2)
        else:
            cv2.line(frame, (line_coord, 0), (line_coord, h), (0, 255, 255), 2)

        for x1, y1, x2, y2 in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        for object_id, centroid in objects.items():
            cv2.circle(frame, tuple(centroid), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        summary = f"Entries: {self.entries}  Exits: {self.exits}  In-store: {self.current_occupancy}"
        cv2.putText(frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return frame

    def _log_event(self, event_type):
        if psycopg2 is None:
            return
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO footfall_events (store_id, event_type, timestamp)
                VALUES (%s, %s, %s)
                """,
                (self.store_id, event_type, datetime.now()),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[DB WARNING] Could not log event: {e}")

    def get_summary(self):
        return {
            "store_id": self.store_id,
            "entries": self.entries,
            "exits": self.exits,
            "current_occupancy": self.current_occupancy,
            "timestamp": datetime.now().isoformat(),
        }

    def run(self, display=True, save_output=None):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(save_output, fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self.process_frame(frame)

                if writer:
                    writer.write(annotated)
                if display:
                    cv2.imshow("RetailVision AI — Footfall Counter", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("Final summary:", self.get_summary())


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RetailVision AI Footfall Counter")
    parser.add_argument("--source", default=0, help="Camera index, RTSP URL, or video file path")
    parser.add_argument("--line-position", type=float, default=0.5, help="Line position fraction (0-1)")
    parser.add_argument("--orientation", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--store-id", default="store_1")
    parser.add_argument("--save", default=None, help="Path to save annotated output video")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    counter = FootfallCounter(
        source=source,
        line_position=args.line_position,
        orientation=args.orientation,
        store_id=args.store_id,
    )
    counter.run(display=not args.no_display, save_output=args.save)
