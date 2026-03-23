"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MnDOT URLs
    camera_image_url: str = "https://video.dot.state.mn.us/video/image/metro/{camera_id}"
    hls_stream_url: str = "https://video.dot.state.mn.us/public/{camera_id}.stream/playlist.m3u8"
    metro_config_url: str = "https://data.dot.state.mn.us/iris_xml/metro_config.xml.gz"
    mayfly_url: str = "https://data.dot.state.mn.us/mayfly"

    # Intervals
    sse_interval_seconds: int = 0  # No extra sleep; 30s window filled by frame collection
    metro_config_refresh_seconds: int = 3600
    mayfly_cache_seconds: int = 60

    # YOLO
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.3
    vehicle_classes: list[int] = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # Spatial matching
    max_match_distance_m: float = 500.0

    # Road geometry (shapefile)
    shapefile_path: str = "data/shp/Trunk_Highways_in_Minnesota.shp"
    road_search_radius_m: float = 300.0

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "sensors"

    # Multi-frame tracking (30s aggregation window: 10 frames × 3s = 30s)
    tracking_frames: int = 10
    tracking_interval_s: float = 3.0

    # OpenAI (VLM ROI generation)
    openai_api_key: str = ""
    openai_model: str = "gpt-5.3-chat-latest"
    roi_data_dir: str = "data/rois"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "C2D_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
