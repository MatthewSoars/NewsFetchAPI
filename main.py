import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="debug",  # Set log level to 'info' or 'debug' for more verbosity
        reload=True,       # Enables auto-reload in development mode
    )
