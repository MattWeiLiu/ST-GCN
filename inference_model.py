# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import joblib
from configuration import Config
from base_model import UNetSTMGCN2
from load_data import load_adj_npz


class InferenceModel(object):

#     def __init__(self, config_path="config_inference.yaml"):
#         self.config = Config(os.path.join(os.getcwd(), config_path))
    def __init__(self, config):
        self.config = config
        self.scalar = joblib.load(self.config.inference.scalar_path)

        # Set GPU as available physical device
        my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')[0]
        tf.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
        print("GPU setting is finished.")

        # Construct heterogeneous graph
        nbhd_adj, simi_adj, cont_adj = load_adj_npz(npz_nbhd_path=self.config.data.nbhd_adj_path,
                                                    npz_simi_path=self.config.data.simi_adj_path,
                                                    npz_cont_path=self.config.data.cont_adj_path)
        print("Heterogeneous graph construction is finished.")

        # Initialize inference model
        self.model = UNetSTMGCN2(batch_size=self.config.inference.batch_size,
                                 num_vertices=self.config.graph.num_vertices,
                                 time_length=self.config.data.time_length,
                                 feature_dims=self.config.data.feature_dims,
                                 hidden_dims=self.config.model.hidden_dims,
                                 en_ksize=self.config.model.en_ksize,
                                 de_ksize=self.config.model.de_ksize,
                                 nbhd_adj=nbhd_adj,
                                 simi_adj=simi_adj,
                                 cont_adj=cont_adj,
                                 with_attention=self.config.model.attention,
                                 regularizer_scale=self.config.model.regularizer_scale)
        _ = self.model(tf.random.uniform(shape=[self.config.inference.batch_size, 16, 11520, 3]))
        self.model.load_weights(self.config.inference.model)
        print("Inference model initialization is finished.")

    def scale_transform(self, x):
        x = x.transpose((0, 1, 3, 2))
        x = x.reshape((self.config.inference.batch_size * 16 * 3, 11520))
        x = self.scalar.transform(x)
        x = x.reshape((self.config.inference.batch_size, 16, 3, 11520))
        return x.transpose((0, 1, 3, 2))

    def inference(self, input):
        return self.scalar.inverse_transform(
            self.model(
                tf.convert_to_tensor(self.scale_transform(input), dtype=tf.float32)
            )
        )
