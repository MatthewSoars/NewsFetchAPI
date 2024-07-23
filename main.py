import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Enables auto-reload in development mode
        workers=1  # Ensures only one worker is used
    )
