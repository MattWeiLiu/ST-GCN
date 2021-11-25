import os
from tqdm import tqdm
from configuration import Config
import pandas as pd
import numpy as np
from load_data import generate3feat
from datetime import timedelta, datetime
from google.cloud import bigquery
from inference_model import InferenceModel
import logging
import pytz

## Time Zone ##
tw = pytz.timezone('Asia/Taipei')


#
# Configuration
# ----------------------------------------------------------------------------------------------------------------------
config = Config(os.path.join(os.getcwd(), "config_inference.yaml"))

class Inference(object):
    
    def __init__(self, config):
        self.infer_model = InferenceModel(config)
        self.form = config.inference.form
        self.sorting = pd.read_csv(config.data.geohash_path, header=None)
        self.ratio = 1

    #
    # Inference Data Loading
    # ----------------------------------------------------------------------------------------------------------------------
    def query(self, time, ret=False, dest=None, progress=False): 
        ## bigquery path
        project_id = config.inference.bigquery.project_id
        dataset_id = config.inference.bigquery.dataset_id
        table_id   = config.inference.bigquery.table_id

        ## query
        sql = 'select distinct * from `{}.{}.{}` where Time in (\'{}\')'.format(project_id, dataset_id, table_id, '\', \''.join(time))
        bqclient = bigquery.Client()
        cfg = bigquery.QueryJobConfig()
        cfg.use_query_cache = False
        resp = bqclient.query(sql, job_config=cfg)
        df = pd.DataFrame(list(map(lambda row: dict(row), resp.result(timeout=60))))
        return df

    def preprocessing(self, df, Bin):
        df = df.reindex(columns=['Time', 'Vertex','Count'])
        df = df.pivot(index='Time', columns='Vertex', values='Count')
        
        if df.shape[0] != 48:
            logging.warn(str(set(pd.to_datetime(Bin)) - set(df.index)))
        
        df = df.reindex(columns=self.sorting[0].values).fillna(0)
        if df.shape[0] == 0:
            df = df.reindex(index=sorted(pd.to_datetime(Bin)))
            df = df.fillna(0)
        else: 
            df = df.reindex(index=sorted(pd.to_datetime(Bin)))
            df = df.interpolate(axis=0, limit_direction='both')
            
        x = df.values.reshape(-1, 3, 11520).sum(1)
        x = np.array(x, dtype=np.float32)
        x = np.expand_dims(x, axis=[0,1])
        x = x.transpose(0, 2, 3, 1)
        return x
    
    def __call__(self, ts):
        #
        # Query filter
        # per 15 min
        q_time = [(datetime.fromtimestamp(ts, tz=tw) - timedelta(minutes=q*5)).strftime(self.form)  for q in range(16*3)]

        #
        # per day
        d_time = []
        for q in range(16):
            d_time.extend([(datetime.fromtimestamp(ts, tz=tw) - timedelta(minutes=24*60*(q+1)-5*(n+1))).strftime(self.form) for n in range(3)])

        #
        # per week
        w_time = []
        for q in range(16):
            w_time.extend([(datetime.fromtimestamp(ts, tz=tw) - timedelta(minutes=24*60*7*(q+1)-5*(n+1))).strftime(self.form) for n in range(3)])
            
        
        df_q = self.query(q_time)
        X_Q = self.preprocessing(df_q, q_time)

        df_d = self.query(d_time)
        X_day = self.preprocessing(df_d, d_time)

        df_w = self.query(w_time)
        X_week = self.preprocessing(df_w, w_time)

        X = np.concatenate([X_Q, X_day, X_week], axis=3)
        if config.valid.coefficient:
            self.ratio = X_day[:, -config.valid.coefficient:, :, :].mean() / X_day.mean()
        y_pred_unscaled = self.infer_model.inference(X)
        y_pred_unscaled *= self.ratio
        res = pd.Series(y_pred_unscaled[0,:], index=self.sorting[0].values).to_json()
        return res
