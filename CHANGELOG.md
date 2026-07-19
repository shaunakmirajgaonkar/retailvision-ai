# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-07-19
### Added
- Shelf inventory monitoring module (`shelf_monitor.py`) with classical-CV
  fallback fill-ratio detector and hook for custom-trained product detection models
- `shelf_snapshots` table and `/shelf/latest`, `/shelf/history` API endpoints
- Streamlit dashboard (`dashboard.py`) showing live occupancy, hourly footfall
  chart, and shelf status per zone

## [1.0.0] - 2026-07-19
### Added
- Customer counting & footfall analysis module (`footfall_counter.py`) using
  YOLOv8 person detection and a lightweight centroid tracker
- Line-crossing entry/exit counting with live occupancy tracking
- PostgreSQL schema (`schema.sql`) for footfall events and hourly rollups
- FastAPI service (`api.py`) exposing footfall summary and hourly endpoints
- Initial README with setup instructions
