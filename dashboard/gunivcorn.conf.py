# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 100
max_requests = 100
max_requests_jitter = 100
timeout = 30
keepalive = 5