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

    # Multi-frame tracking (legacy JPEG snapshot mode)
    tracking_frames: int = 10
    tracking_interval_s: float = 3.0

    # HLS stream processing
    hls_enabled: bool = True
    hls_target_fps: float = 15.0
    hls_reconnect_delay_s: float = 5.0
    hls_reconnect_max_retries: int = 10
    aggregation_window_s: float = 30.0

    # YOLO tracking (ByteTrack)
    yolo_tracker: str = "bytetrack.yaml"

    # GPU management
    max_concurrent_streams: int = 4

    # Speed estimation: pixels-per-meter calibration (camera-dependent)
    # Default ~7.0 is rough estimate for typical MnDOT highway cameras at 720x480
    speed_calibration_ppm: float = 7.0

    # OpenAI (VLM ROI generation)
    openai_api_key: str = ""
    openai_model: str = "gpt-5.3-chat-latest"
    roi_data_dir: str = "data/rois"

    # Camera movement detection
    movement_ssim_threshold: float = 0.60
    reference_frame_dir: str = "data/reference_frames"

    # Camera calibration defaults (Phase 1B: geometric projection)
    default_camera_height_m: float = 9.1  # ~30 ft typical MnDOT pole mount
    default_camera_fov_deg: float = 60.0  # Horizontal FOV
    default_camera_tilt_deg: float = 25.0  # Downward tilt from horizontal
    calibration_min_vehicles: int = 50
    calibration_data_dir: str = "data/calibrations"

    # Trajectory-based ROI clustering
    cluster_min_trajectories: int = 200   # vehicles needed for auto-generation
    cluster_min_force: int = 50           # minimum for forced (API) generation
    cluster_min_group_size: int = 20      # min trajectories per road-direction group
    cluster_max_angle_deg: float = 45.0   # max circular distance for direction assignment
    cluster_simplify_px: float = 10.0     # Shapely simplify tolerance (pixels)

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "C2D_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
