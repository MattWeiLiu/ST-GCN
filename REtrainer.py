# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm

from configuration import Config
from base_model import UNetSTMGCN2
import joblib
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = joblib.load('scaler.pkl')
from load_data import generate3feat, load_adj_npz, get_batch

#
# Configuration Loading
# ----------------------------------------------------------------------------------------------------------------------
config = Config(os.path.join(os.getcwd(), "config_local_3Feat.yaml"))

# Set GPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')[0]
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='GPU')


# Load data
# X.shape = [batch_size, time_length, num_vertices, feature_dims]
# y.shape = [batch_size, num_vertices]

def load_data_3feat(path_data, time_steps, num_vertices, feature_dims, nrows=None, skiprows=0, normalize_type='vertex'):
    # load dataset
    
    # dataset.shape = [time_length, num_vertices, feature_dims]
    if path_data.endswith('.csv'):
        dataset = pd.read_csv(path_data, index_col=0, nrows=nrows, skiprows=skiprows)
        dataset = dataset.values.astype('float32')
    else:
        dataset = np.load(path_data, allow_pickle=True)['arr_0']
        dataset = dataset[:nrows,:,:feature_dims]
        dataset = dataset.astype('float32')
    
    # normalize features
    # scaled = scaler.fit_transform(dataset)
    scaled = scaler.transform(dataset)
    
    print('dataset.shape = {} -------------------------------------------------------------------'.format(dataset.shape))
    
    X_test,  y_test  = generate3feat(scaled,  time_steps, num_vertices, feature_dims)
    return X_test,  y_test

# ----------------------------------------------------------------------------------------------------------------------
nbhd_adj, simi_adj, cont_adj = load_adj_npz(npz_nbhd_path=config.data.nbhd_adj_path,
                                            npz_simi_path=config.data.simi_adj_path,
                                            npz_cont_path=config.data.cont_adj_path)
X_test, y_test= load_data_3feat(config.inference.data_path, 
                                time_steps=config.data.time_length, 
                                num_vertices=config.graph.num_vertices, 
                                feature_dims=config.data.feature_dims,
                                normalize_type=config.train.normalize_type,  
                                skiprows=0)
# print('X_train:{}, y_train:{}'.format(X_train.shape, y_train.shape))
# print('X_val:{}, y_val:{}'.format(X_val.shape, y_val.shape))
print('X_test:{}, y_test:{}'.format(X_test.shape, y_test.shape))

#
# Create model
# ----------------------------------------------------------------------------------------------------------------------
model = UNetSTMGCN2(batch_size=config.train.batch_size,
                    num_vertices=config.graph.num_vertices, 
                    time_length=config.data.time_length, 
                    feature_dims=config.data.feature_dims, 
                    hidden_dims=config.model.hidden_dims,
                    en_ksize=config.model.en_ksize, 
                    de_ksize=config.model.de_ksize,
                    nbhd_adj=nbhd_adj, 
                    simi_adj=simi_adj, 
                    cont_adj=cont_adj, 
                    with_attention=config.model.attention,
                    regularizer_scale=config.model.regularizer_scale)

_ = model(tf.convert_to_tensor(X_test[:config.train.batch_size]))
model.load_weights(config.inference.model)

optimizer = tf.keras.optimizers.Adam(learning_rate=config.optimizer.learning_rate)
train_batch_number = int(X_test.shape[0] / config.train.batch_size)

#
# Train Model
# ----------------------------------------------------------------------------------------------------------------------
train_losses = []
best_loss = 1000

print('Model name: {}'.format(model.name))
for e in range(config.train.epoch):
    train_loss_cache = []
    for _ in tqdm(range(train_batch_number), total=train_batch_number):
        x_batch, y_batch = get_batch(X=X_test, y=y_test, batch_size=config.train.batch_size)

        with tf.GradientTape() as tape:
            y_pred = model(tf.convert_to_tensor(x_batch))
            train_loss = tf.keras.losses.MSE(y_batch, y_pred)

        gradients = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss_cache.append(train_loss.numpy())

    res_variance_unscaled = []
    res_val_losses = []

    train_loss_epoch = np.mean(train_loss_cache)
    train_losses.append(train_loss_epoch)
#     if train_loss_epoch < best_loss:
#         model.save_weights('BestModel/MFeat_last1y_16_T3-h32-G1diff_UNetSTMGCN2_mse_opt-adam_bs2_train_acc_15_1_epidemic.h5')
#         best_loss = train_loss_epoch
    print('Epoch: {}/{}\ttrain_loss: {:.6f}'.
          format(e + 1, config.train.epoch, train_loss_epoch))

