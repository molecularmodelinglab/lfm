import json
import os
from celery import Celery


if "REDIS_HOST" in os.environ:
    print("Using local Redis connection for Celery")
    redis_host = os.environ["REDIS_HOST"]
    redis_port = os.environ["REDIS_PORT"]

    celery_app = Celery(
        "tasks",
        broker=f"redis://{redis_host}:{redis_port}/0",
    )

else:
    print("Using SQS connection for Celery")
    sqs_params = json.load(open("secrets/aws.json"))
    broker_url = f"sqs://{sqs_params['key']}:{sqs_params['secret']}@"

    celery_app = Celery(
        "tasks",
        broker=broker_url,
    )

celery_app.conf.update(worker_heartbeat=60)
# celery_app.autodiscover_tasks(["common", "pmf_net"])
celery_app.conf.broker_transport_options = {"visibility_timeout": 43200}