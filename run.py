"""Entry point for the Camera2Detector dashboard."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("C2D_PORT", "30000")),
        reload=os.environ.get("C2D_RELOAD", "0") == "1",
    )
