import os
import uvicorn

if __name__ == "__main__":
    total_workers = int(os.getenv("TOTAL_WORKERS", "1"))  # Set default to 1 worker
    worker_index = int(os.getenv("WORKER_INDEX", "0"))    # Set default to worker 0

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=total_workers
    )
