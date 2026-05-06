# celery_config.py - Centralized Celery app instance
from celery import Celery

# Define your broker and backend URLs
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

# Create the Celery app instance
celery_app = Celery(
    'eduPy',
    broker=broker_url,
    backend=result_backend,
    include=['celery_worker'] # <-- Add your new module here
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Kolkata',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        'max_retries': 10,
        'interval_start': 0,
        'interval_step': 0.5,
        'interval_max': 3,
        'visibility_timeout': 18000 # 5 hours
    }
)
