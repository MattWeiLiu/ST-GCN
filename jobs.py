import os
import json
import datetime
import time
import logging
from inference import Inference
from configuration import Config


class JobManager:
    redis = None
    bq = None
    bqConfig = None

    def __init__(self, redis_conn, bq_conn, bqConf):
        self.redis = redis_conn
        self.bq = bq_conn
        self.bqConfig = bqConf

    def infer_send_car_job(self, id, dt):
        # Check ID duplicated TODO

        # Get timestamp floor to 5 mins
        t = floor_timestamp_to_minutes(dt, 5)
        timestamp_key = int(t.timestamp()) * 1000
        logging.debug('job start timestamp_key: %d' % timestamp_key)

        # Infer results by timestamp
        # global config
        logging.debug('job before inferring timestamp_key: %d' % timestamp_key)
        inferer = Inference(self.bqConfig)
        logging.debug('job after inferring timestamp_key: %d' % timestamp_key)
        res_json_str = inferer(float(t.timestamp()))
        res_dict = json.loads(res_json_str)

        # Store results to redis
        self.store_cache(timestamp_key, res_json_str)

        # Store results to bigquery
        self.store_result(t.timestamp(), res_dict)

        logging.info('job end timestamp_key: %d' % timestamp_key)
        return t.timestamp()*1000, res_dict

    # store result to Bigquery

    def store_result(self, timepoint, res):
        curr_time = time.time()
        dataset_id = self.bqConfig.inference.bigquery.dataset_id
        table_id = self.bqConfig.inference.bigquery.result_table_id

        # Prepare insert rows
        rows = [[timepoint, k, v, curr_time] for k, v in res.items()]

        # Bigquery insert_rows max limit 10000 rows
        batch = 5000
        for i in range(0, len(rows), batch):
            table_ref = self.bq.dataset(dataset_id).table(table_id)
            table = self.bq.get_table(table_ref)
            print(table)
            errors = self.bq.insert_rows(
                table_ref, rows[i:i + batch], table.schema)
            if errors != []:
                raise Exception('Bigquery insert failed %s' % str(errors[0]))
        return

    def store_cache(self, timestamp_key, result):
        try:
            self.redis.set(timestamp_key, result)
            # Expire the key after 12 hr
            self.redis.expire(timestamp_key, 60 * 60 * 12)
        except Exception as e:
            raise Exception('Redis store cache error: ' + str(e))


def floor_timestamp_to_minutes(t, min):
    return t - datetime.timedelta(minutes=t.minute %
                                  min, seconds=t.second, microseconds=t.microsecond)
