# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tqdm import tqdm

from configuration import Config
from base_model import UNetSTMGCN2, ImprvSTMGCN
from load_data import *

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
# ----------------------------------------------------------------------------------------------------------------------
nbhd_adj, simi_adj, cont_adj = load_adj_npz(npz_nbhd_path=config.data.nbhd_adj_path,
                                            npz_simi_path=config.data.simi_adj_path,
                                            npz_cont_path=config.data.cont_adj_path)
X_train, y_train, X_val, y_val = load_data_3feat(config.data.data_path,
                                                 time_steps=config.data.time_length,
                                                 num_vertices=config.graph.num_vertices,
                                                 feature_dims=config.data.feature_dims,
                                                 normalize_type=config.train.normalize_type,  
                                                 skiprows=0, 
                                                 nrows=46752)
print('X_train:{}, y_train:{}'.format(X_train.shape, y_train.shape))
print('X_val:{}, y_val:{}'.format(X_val.shape, y_val.shape))


#
# Create model
# ----------------------------------------------------------------------------------------------------------------------

## UNets
model = UNetSTMGCN2(batch_size=config.train.batch_size, num_vertices=config.graph.num_vertices, 
                    time_length=config.data.time_length, feature_dims=config.data.feature_dims, 
                    hidden_dims=config.model.hidden_dims,
                    en_ksize=config.model.en_ksize, de_ksize=config.model.de_ksize,
                    nbhd_adj=nbhd_adj, simi_adj=simi_adj, cont_adj=cont_adj, 
                    with_attention=config.model.attention,
                    regularizer_scale=config.model.regularizer_scale)
### 
# model(tf.convert_to_tensor(X_train[:config.train.batch_size]))
# model.load_weights('MFeat_last1y_16_T3-h8-G1diff_UNetSTMGCN2_mse_opt-adam_bs32_acc_valid.h5')
### 

if config.optimizer.method == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.optimizer.learning_rate)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.optimizer.learning_rate)
train_batch_number = int(X_train.shape[0] / config.train.batch_size)
val_batch_number = int(X_val.shape[0] / config.train.batch_size)


#
# Train Model
# ----------------------------------------------------------------------------------------------------------------------
train_losses = []
valid_losses = []
valid_MSEs = []
valid_RMSEs = []
best_train_loss = 1000
best_valid_loss = 1000
print('Model name: {}'.format(model.name))
for e in range(config.train.epoch):
    train_loss_cache = []
    valid_loss_cache = []
    for _ in tqdm(range(train_batch_number), total=train_batch_number):
    # for _ in range(train_batch_number):
        x_batch, y_batch = get_batch(X=X_train, y=y_train, batch_size=config.train.batch_size)

        with tf.GradientTape() as tape:
            y_pred = model(tf.convert_to_tensor(x_batch))
            train_loss = tf.keras.losses.MSE(y_batch, y_pred)
            # train_loss = tf.keras.losses.poisson(y_batch, y_pred)

        gradients = tape.gradient(train_loss, model.trainable_variables)
        # gradients = [(tf.clip_by_value(grad, -0.2, 0.2)) for grad in gradients]    # gradient clipping
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss_cache.append(train_loss.numpy())

    res_variance_unscaled = []
    res_val_losses = []
    for _ in tqdm(range(val_batch_number), total=val_batch_number):
    # for _ in range(val_batch_number):
        x_batch_val, y_batch_val = get_batch(X=X_val, y=y_val, batch_size=config.train.batch_size)

        y_pred_val = model(tf.convert_to_tensor(x_batch_val))
        val_loss = tf.keras.losses.MSE(y_batch_val, y_pred_val)
        # val_loss = tf.keras.losses.poisson(y_batch_val, y_pred_val)
        # tf.nn.log_poisson_loss
        valid_loss_cache.append(val_loss.numpy())

        if config.train.normalize_type == 'all':
            y_pred_val = y_pred_val.numpy()
            y_pred_val_unscaled = scaler.inverse_transform(y_pred_val.reshape((-1, 1))).reshape(y_pred_val.shape)
            y_batch_val_unscaled = scaler.inverse_transform(y_batch_val.reshape((-1, 1))).reshape(y_batch_val.shape)
        else:
            y_pred_val_unscaled = scaler.inverse_transform(y_pred_val)
            y_batch_val_unscaled = scaler.inverse_transform(y_batch_val)

        res_val_losses.append(val_loss)
        variance_one_batch = np.mean((y_pred_val_unscaled - y_batch_val_unscaled) ** 2, 1)
        res_variance_unscaled.append(variance_one_batch)

    train_loss_epoch = np.mean(train_loss_cache)
    valid_loss_epoch = np.mean(valid_loss_cache)
    mse = np.mean(res_variance_unscaled)
    rmse = np.sqrt(mse)

    train_losses.append(train_loss_epoch)
    valid_losses.append(valid_loss_epoch)
    valid_MSEs.append(mse)
    valid_RMSEs.append(rmse)
    
    if train_loss_epoch < best_train_loss:
        term_model = str(type(model)).split("'")[1].split(".")[1]
        term_nfeat = 'SFeat' if config.data.feature_dims == 1 else 'MFeat'
        term_length = config.data.time_length
        term_attn = 'Atn-' if config.model.attention else ''
        term_lyrs = len(config.model.en_ksize)
        term_hdims = config.model.hidden_dims
        term_ndiff = config.model.max_diffusion_step
        term_loss = config.model.loss
        term_opt = config.optimizer.method
        term_bs = config.train.batch_size
        model.save_weights('{}_last1y_{}_{}T{}-h{}-G{}diff_{}_{}_opt-{}_bs{}_train.h5'.
                           format(term_nfeat, term_length, term_attn, term_lyrs, 
                                  term_hdims, term_ndiff, term_model, term_loss, term_opt, term_bs))
        best_train_loss = train_loss_epoch
    print('Epoch: {}/{}\ttrain_loss: {:.6f}\tval_loss: {:.6f}'.
          format(e + 1, config.train.epoch, train_loss_epoch, valid_loss_epoch))
    print('Val Set Performance: Unscaled MSE: {} Unscaled RMSE: {}'.format(mse, rmse))
