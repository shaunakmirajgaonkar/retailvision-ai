-- RetailVision AI — Footfall & Customer Counting Schema

CREATE TABLE IF NOT EXISTS footfall_events (
    id SERIAL PRIMARY KEY,
    store_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(10) NOT NULL CHECK (event_type IN ('entry', 'exit')),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_footfall_store_time
    ON footfall_events (store_id, timestamp);

-- Hourly rollup view for the dashboard
CREATE OR REPLACE VIEW footfall_hourly AS
SELECT
    store_id,
    date_trunc('hour', timestamp) AS hour,
    COUNT(*) FILTER (WHERE event_type = 'entry') AS entries,
    COUNT(*) FILTER (WHERE event_type = 'exit') AS exits
FROM footfall_events
GROUP BY store_id, date_trunc('hour', timestamp)
ORDER BY hour;

-- Shelf inventory monitoring
CREATE TABLE IF NOT EXISTS shelf_snapshots (
    id SERIAL PRIMARY KEY,
    store_id VARCHAR(50) NOT NULL,
    zone_name VARCHAR(100) NOT NULL,
    fill_ratio REAL NOT NULL,
    product_count INTEGER,
    status VARCHAR(20) NOT NULL CHECK (status IN ('stocked', 'low_stock')),
    method VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shelf_store_time
    ON shelf_snapshots (store_id, zone_name, timestamp);

