# coding=utf-8

from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError
import json
import os
import redis
from twtaxi_requests import InferSendCarRequest
from twtaxi_responses import InferSendCarResponse
from jobs import JobManager
from google.cloud import bigquery
from inference import Inference
from configuration import Config
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging
from datetime import datetime


# Cloud logger setup
client = google.cloud.logging.Client()
handler = CloudLoggingHandler(
    client, name=os.getenv("LOGGER_NAME", "inferring-server"))
logging.getLogger().setLevel(logging.INFO)
setup_logging(handler)

# Bigquery setup
bq_config = Config(os.path.join(os.getcwd(), "config_inference.yaml"))
bq_config.inference.bigquery.project_id = os.getenv(
    'PROJECT_ID', bq_config.inference.bigquery.project_id)
bq_config.inference.bigquery.dataset_id = os.getenv(
    'GBQ_DATASET_ID', bq_config.inference.bigquery.dataset_id)
bq_config.inference.bigquery.table_id = os.getenv(
    'GBQ_TABLE_ID', bq_config.inference.bigquery.table_id)
bq_config.inference.bigquery.result_table_id = os.getenv(
    'GBQ_RESULT_TABLE_ID', bq_config.inference.bigquery.result_table_id)

# Register Bigquery, Redis services
jobMgr = JobManager(redis.Redis(host=os.getenv(
    'REDIS_HOST', '10.233.169.107'), port=6379, db=0), bigquery.Client(), bq_config)


def infer_send_car_demand_cb(message):
    # logging.info("Received message: {}".format(message))
    # logging.info("Received job: message_id {}, publish_time UTC: {} , target_time: {}".format(
    #     message.message_id, message.publish_time, infer_time))
    try:
        if message.attributes.get('infer_time') is None:
            logging.info("Use publish_time")
            target_time = message.publish_time
        else:
            logging.info("Use infer_time")
            infer_time = message.attributes.get('infer_time')
            target_time = datetime.fromtimestamp(infer_time)
            
        logging.info("Target time: {}".format(target_time))
        jobMgr.infer_send_car_job(message.message_id, target_time)
        message.ack()
        logging.debug('Job done id: %s' % message.message_id)
    except Exception as e:
        logging.exception(e)


# Polling inferring job
def pollingJobs():
    timeout = 300
    flow_control = pubsub_v1.types.FlowControl(max_messages=1)
    while True:
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(os.getenv(
            'PROJECT_ID', 'taiwan-taxi'), os.getenv('INFERRING_DATA_UPDATE_TOPIC', 'infer-send-car-demand'))
        streaming_pull_future = subscriber.subscribe(
            subscription_path, callback=infer_send_car_demand_cb, flow_control=flow_control)
        logging.info(
            "Listening for messages on {}..\n".format(subscription_path))

        with subscriber:
            try:
                streaming_pull_future.result(timeout=timeout)
            except TimeoutError:
                streaming_pull_future.cancel()


if __name__ == "__main__":
    pollingJobs()
