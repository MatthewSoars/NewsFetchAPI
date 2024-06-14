import multiprocessing

# Define the number of worker processes (generally, 2-4 x $num_cores)
workers = multiprocessing.cpu_count() * 2 + 1
# Set the worker class to 'uvicorn.workers.UvicornWorker'
worker_class = "uvicorn.workers.UvicornWorker"
# Bind to the appropriate address and port
bind = "0.0.0.0:8080"
# Increase the timeout if needed (default is 30 seconds)
timeout = 120
# Optional: set log level
loglevel = "info"
# Optional: define a log file
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"