# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import scipy.sparse as sp
import numpy as np

from gconv import MGConv, calculate_scaled_laplacian
from unet_model import TempUNet


class HierarchicalConv1D(keras.Model):

    def __init__(self, input_dims, num_vertices,
                 time_length=12, num_layers=3, hidden_dims=16, channels_expand=3, regularizer_scale=0.0003):
        super(HierarchicalConv1D, self).__init__()

        self.input_dims = input_dims
        self.num_vertices = num_vertices
        self.time_length = time_length
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.regularizer_scale = regularizer_scale
        self.ch_expnd = channels_expand
        layers = []
        for layer_index in range(self.num_layers):
            filters_num = self.input_dims * self.ch_expnd ** (layer_index + 1)
            layers += [keras.layers.Conv1D(filters_num, padding='same', kernel_size=3, activation='linear'),
                       keras.layers.BatchNormalization(),
                       keras.layers.ELU(),
                       keras.layers.Conv1D(filters_num, padding='same', kernel_size=3, activation='linear'),
                       keras.layers.BatchNormalization(),
                       keras.layers.ELU(),
                       keras.layers.MaxPool1D(strides=2),
                       ]
        self.net_CNNs = tf.keras.Sequential(layers)
        self.net_Dropout = keras.layers.Dropout(0.1)
        self.net_Flatten = tf.keras.layers.Flatten()
        self.net_FC1 = tf.keras.layers.Dense(units=self.hidden_dims, activation='tanh',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))

    def __call__(self, x):
        batch_size, time_length, num_vertices, feature_dims = x.get_shape()
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch_size, num_vertices, time_length, feature_dims]
        x = tf.reshape(x, [-1, time_length, feature_dims])  # [batch_size * num_vertices, time_length, hidden_dims]
        x = self.net_CNNs(x)  # [batch_size * num_vertices, hidden_dims] 
        x = self.net_Dropout(x)
        x = self.net_Flatten(x)
        x = self.net_FC1(x)
        cnn_output = tf.reshape(x, [batch_size, num_vertices, self.hidden_dims])
        return cnn_output


class HierarchicalRNN(keras.Model):

    def __init__(self, num_layers=3, hidden_dims=16):
        super(HierarchicalRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        rnn_layers = [tf.keras.layers.GRU(self.hidden_dims, return_sequences=True) for _ in range(self.num_layers - 1)]
        rnn_layers += [tf.keras.layers.GRU(self.hidden_dims)]
        self.net_RNNs = tf.keras.Sequential(rnn_layers)

    def __call__(self, x):
        batch_size, time_length, num_vertices, feature_dims = x.get_shape()
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch_size, num_vertices, time_length, feature_dims]
        x = tf.reshape(x, (-1, time_length, feature_dims))
        x = self.net_RNNs(x)  # [batch_size * num_vertices, hidden_dims]
        rnn_output = tf.reshape(x, [batch_size, num_vertices, self.hidden_dims])

        return rnn_output


class GraphConvOp(keras.layers.Layer):

    def __init__(self, L_mx, num_vertices, batch_size, time_length, feat_dim):
        super(GraphConvOp, self).__init__()
        self.L_mx = L_mx
        self.num_vertices = num_vertices
        self.batch_size = batch_size
        self.time_length = time_length
        self.feat_dim = feat_dim

    def __call__(self, inputs):
        # inputs.shape = [batch_size, time_length, num_vertices, feat_dim]
        inputs = tf.transpose(inputs, perm=[2, 3, 0, 1])  # (num_vertices, feat_dim, batch_size, time_length)
        inputs = tf.reshape(inputs, shape=[self.num_vertices, -1])  # (num_vertices, feat_dim*batch_size*time_length)
        outputs = tf.sparse.sparse_dense_matmul(self.L_mx, inputs)  # (num_vertices, feat_dim*batch_size*time_length)
        outputs = tf.reshape(outputs, shape=[self.num_vertices, self.feat_dim, self.batch_size, self.time_length])
        return tf.transpose(outputs, perm=[2, 3, 0, 1])


class ContextGate(keras.Model):
    def __init__(self, L_mx, batch_size, time_length, num_vertices, feature_dims, regularizer_scale=0.003):
        super(ContextGate, self).__init__()
        self.num_vertices = num_vertices
        self.batch_size = batch_size
        self.time_length = time_length
        self.feature_dims = feature_dims
        self.regularizer_scale = regularizer_scale

        self.GraphConvOp = GraphConvOp(L_mx, self.num_vertices, self.batch_size, self.time_length, self.feature_dims)

        self.net_CntxtGate = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.feature_dims, activation='elu',
                                  kernel_regularizer=keras.regularizers.l2(self.regularizer_scale)),
            tf.keras.layers.Dense(units=1, activation='sigmoid',
                                  kernel_regularizer=keras.regularizers.l2(self.regularizer_scale))])

    def __call__(self, inputs):
        # Contextual gated by average pooling along with vertex axis from Eq. (6)
        # [batch_size, time_length, num_vertex, feature_dim]
        avg_pool = tf.concat([tf.math.reduce_mean(inputs, axis=2),
                              tf.math.reduce_mean(self.GraphConvOp(inputs), axis=2)], axis=-1)
        # [batch_size, time_length, feature_dim*2]
        avg_pool = tf.reshape(avg_pool, [self.batch_size * self.time_length, self.feature_dims * 2])
        # [batch_size*time_length, feature_dim*2]
        avg_pool = self.net_CntxtGate(avg_pool)
        # [batch_size*time_length, 1]
        avg_pool = tf.expand_dims(tf.reshape(avg_pool, (self.batch_size, self.time_length, 1)), 2)
        # [batch_size, time_length, 1, 1]
        return tf.math.multiply(inputs, avg_pool)


class NoOpGate(keras.Model):
    def __call__(self, inputs):
        return inputs


class STMGCN(keras.Model):

    def __init__(self, batch_size, num_vertices, time_length, feature_dims, hidden_dims,
                 num_layers, nbhd_adj, simi_adj, cont_adj, with_attention=True,
                 extractor_type='rnn', regularizer_scale=0.003, lambda_max=2):
        super(STMGCN, self).__init__()

        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.time_length = time_length
        self.hidden_dims = hidden_dims
        self.feature_dims = feature_dims
        self.num_layers = num_layers
        self.regularizer_scale = regularizer_scale
        self.lambda_max = lambda_max
        self.with_attention = with_attention

        def get_sparse_laplacian(adj_mx):
            L_mx = calculate_scaled_laplacian(adj_mx, lambda_max=self.lambda_max)
            row, col, val = sp.find(L_mx)
            indices = np.column_stack((row, col))
            return tf.sparse.SparseTensor(indices, val, L_mx.shape)

        self.L_nbhd = get_sparse_laplacian(nbhd_adj)
        self.L_simi = get_sparse_laplacian(simi_adj)
        self.L_cont = get_sparse_laplacian(cont_adj)

        if self.with_attention:
            self.net_CntxtGate_nbhd = ContextGate(self.L_nbhd, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_simi = ContextGate(self.L_simi, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_cont = ContextGate(self.L_cont, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
        else:
            self.net_CntxtGate_nbhd = NoOpGate()
            self.net_CntxtGate_simi = NoOpGate()
            self.net_CntxtGate_cont = NoOpGate()

        if extractor_type == 'rnn':
            self.net_tempExtractor = HierarchicalRNN(num_layers=self.num_layers, hidden_dims=self.hidden_dims)
        else:
            self.net_tempExtractor = HierarchicalConv1D(input_dims=self.feature_dims, num_vertices=self.num_vertices,
                                                        time_length=self.time_length, num_layers=self.num_layers,
                                                        channels_expand=self.channels_expand,
                                                        hidden_dims=self.hidden_dims,
                                                        regularizer_scale=self.regularizer_scale)

        self.net_MGCN_nbhd = MGConv(self.hidden_dims, {'nbhd': self.L_nbhd}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_simi = MGConv(self.hidden_dims, {'simi': self.L_simi}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_cont = MGConv(self.hidden_dims, {'cont': self.L_cont}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)

        self.net_FC1 = tf.keras.layers.Dense(units=16, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))
        self.net_FC2 = tf.keras.layers.Dense(units=1, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))

    def __call__(self, x):

        # X.shape = [batch_size, time_length, num_vertices, feature_dims]

        def contxt_gate_rnn(inputs, net_CntxtGate, net_MGCN):
            gated_inputs = net_CntxtGate(inputs)
            temp_ext_out = self.net_tempExtractor(gated_inputs)  # [batch_size, num_vertices, hidden_dims]
            network_out = net_MGCN(temp_ext_out)
            return network_out

        # output shape: [batch_size, num_vertices, ]
        network_out_nbhd = contxt_gate_rnn(x, self.net_CntxtGate_nbhd, self.net_MGCN_nbhd)
        network_out_simi = contxt_gate_rnn(x, self.net_CntxtGate_simi, self.net_MGCN_simi)
        network_out_cont = contxt_gate_rnn(x, self.net_CntxtGate_cont, self.net_MGCN_cont)

        # Aggregation of the multiple GCN outputs
        network_output = tf.concat([network_out_nbhd, network_out_simi, network_out_cont], axis=-1)
        all_output = self.net_FC2(self.net_FC1(network_output))
        all_output = tf.reshape(all_output, shape=[self.batch_size, self.num_vertices])  # [batch_size, num_vertices]

        return all_output


class STMGCN2(keras.Model):

    def __init__(self, batch_size, num_vertices, time_length, channels_expand, feature_dims, hidden_dims,
                 num_layers, nbhd_adj, simi_adj, cont_adj, with_attention=True,
                 extractor_type='rnn', regularizer_scale=0.003, lambda_max=2):
        super(STMGCN2, self).__init__()

        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.time_length = time_length
        self.channels_expand = channels_expand
        self.hidden_dims = hidden_dims
        self.feature_dims = feature_dims
        self.num_layers = num_layers
        self.regularizer_scale = regularizer_scale
        self.lambda_max = lambda_max
        self.with_attention = with_attention

        def get_sparse_laplacian(adj_mx):
            L_mx = calculate_scaled_laplacian(adj_mx, lambda_max=self.lambda_max)
            row, col, val = sp.find(L_mx)
            indices = np.column_stack((row, col))
            return tf.sparse.SparseTensor(indices, val, L_mx.shape)

        self.L_nbhd = get_sparse_laplacian(nbhd_adj)
        self.L_simi = get_sparse_laplacian(simi_adj)
        self.L_cont = get_sparse_laplacian(cont_adj)

        if self.with_attention:
            self.net_CntxtGate_nbhd = ContextGate(self.L_nbhd, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_simi = ContextGate(self.L_simi, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_cont = ContextGate(self.L_cont, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
        else:
            self.net_CntxtGate_nbhd = NoOpGate()
            self.net_CntxtGate_simi = NoOpGate()
            self.net_CntxtGate_cont = NoOpGate()

        if extractor_type == 'rnn':
            self.net_tempExtractor = HierarchicalRNN(num_layers=self.num_layers, hidden_dims=self.hidden_dims)
        else:
            self.net_tempExtractor = HierarchicalConv1D(input_dims=self.feature_dims, num_vertices=self.num_vertices,
                                                        time_length=self.time_length, num_layers=self.num_layers,
                                                        channels_expand=self.channels_expand,
                                                        hidden_dims=self.hidden_dims,
                                                        regularizer_scale=self.regularizer_scale)

        self.net_MGCN_nbhd = tf.keras.Sequential([
            MGConv(self.hidden_dims, {'nbhd': self.L_nbhd}, 2, self.num_vertices,
                   self.regularizer_scale, lambda_max=lambda_max)])
        self.net_MGCN_simi = tf.keras.Sequential([
            MGConv(self.hidden_dims, {'simi': self.L_simi}, 2, self.num_vertices,
                   self.regularizer_scale, lambda_max=lambda_max)])
        self.net_MGCN_cont = tf.keras.Sequential([
            MGConv(self.hidden_dims, {'cont': self.L_cont}, 2, self.num_vertices,
                   self.regularizer_scale, lambda_max=lambda_max)])

        self.net_FC1 = tf.keras.layers.Dense(units=1, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))

    def __call__(self, x):

        # X.shape = [batch_size, time_length, num_vertices, feature_dims]

        def contxt_gate_rnn(inputs, net_CntxtGate, net_MGCN):
            gated_inputs = net_CntxtGate(inputs)
            temp_ext_out = self.net_tempExtractor(gated_inputs)  # [batch_size, num_vertices, hidden_dims]
            network_out = net_MGCN(temp_ext_out)
            return network_out

        # output shape: [batch_size, num_vertices, ]
        network_out_nbhd = contxt_gate_rnn(x, self.net_CntxtGate_nbhd, self.net_MGCN_nbhd)
        network_out_simi = contxt_gate_rnn(x, self.net_CntxtGate_simi, self.net_MGCN_simi)
        network_out_cont = contxt_gate_rnn(x, self.net_CntxtGate_cont, self.net_MGCN_cont)

        # Aggregation of the multiple GCN outputs
        network_output = tf.concat([network_out_nbhd, network_out_simi, network_out_cont], axis=-1)
        all_output = self.net_FC1(network_output)
        all_output = tf.reshape(all_output, shape=[self.batch_size, self.num_vertices])  # [batch_size, num_vertices]

        return all_output


class TemporalExtractor(keras.Model):

    def __init__(self, batch_size, num_vertices, time_length, feature_dims, hidden_dims,
                 num_layers, channels_expand, extractor_type='rnn', regularizer_scale=0.003):
        super(TemporalExtractor, self).__init__()

        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.time_length = time_length
        self.hidden_dims = hidden_dims
        self.feature_dims = feature_dims
        self.channels_expand = channels_expand
        self.num_layers = num_layers
        self.regularizer_scale = regularizer_scale

        if extractor_type == 'rnn':
            self.net_tempExtractor = HierarchicalRNN(num_layers=self.num_layers, hidden_dims=self.hidden_dims)
        else:
            self.net_tempExtractor = HierarchicalConv1D(input_dims=self.feature_dims, num_vertices=self.num_vertices,
                                                        time_length=self.time_length, num_layers=self.num_layers,
                                                        channels_expand=self.channels_expand,
                                                        hidden_dims=self.hidden_dims,
                                                        regularizer_scale=self.regularizer_scale)

        self.net_FC1 = tf.keras.layers.Dense(units=16, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))
        self.net_FC2 = tf.keras.layers.Dense(units=1, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))

    def __call__(self, x):

        # X.shape = [batch_size, time_length, num_vertices, feature_dims]

        network_output = self.net_tempExtractor(x)
        all_output = self.net_FC2(self.net_FC1(network_output))
        all_output = tf.reshape(all_output, shape=[self.batch_size, self.num_vertices])  # [batch_size, num_vertices]

        return all_output


class ImprvSTMGCN(keras.Model):

    def __init__(self, batch_size, num_vertices, time_length, feature_dims, hidden_dims, # channels_expand,
                 num_layers, nbhd_adj, simi_adj, cont_adj, with_attention=True,
                 extractor_type='rnn', regularizer_scale=0.003, lambda_max=2):
        super(ImprvSTMGCN, self).__init__()

        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.time_length = time_length
        self.hidden_dims = hidden_dims
        # self.channels_expand = channels_expand
        self.feature_dims = feature_dims
        self.num_layers = num_layers
        self.regularizer_scale = regularizer_scale
        self.lambda_max = lambda_max
        self.with_attention = with_attention
        
        def get_sparse_laplacian(adj_mx):
            L_mx = calculate_scaled_laplacian(adj_mx, lambda_max=self.lambda_max)
            row, col, val = sp.find(L_mx)
            indices = np.column_stack((row, col))
            return tf.sparse.SparseTensor(indices, val, L_mx.shape)

        self.L_nbhd = get_sparse_laplacian(nbhd_adj)
        self.L_simi = get_sparse_laplacian(simi_adj)
        self.L_cont = get_sparse_laplacian(cont_adj)

        if self.with_attention:
            self.net_CntxtGate_nbhd = ContextGate(self.L_nbhd, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_simi = ContextGate(self.L_simi, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_cont = ContextGate(self.L_cont, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
        else:
            self.net_CntxtGate_nbhd = NoOpGate()
            self.net_CntxtGate_simi = NoOpGate()
            self.net_CntxtGate_cont = NoOpGate()

        if extractor_type == 'rnn':
            self.net_tempExtractor = HierarchicalRNN(num_layers=self.num_layers, hidden_dims=self.hidden_dims)
        else:
            self.net_tempExtractor = HierarchicalConv1D(input_dims=self.feature_dims, num_vertices=self.num_vertices,
                                                        time_length=self.time_length, num_layers=self.num_layers,
                                                        channels_expand=self.channels_expand,
                                                        hidden_dims=self.hidden_dims,
                                                        regularizer_scale=self.regularizer_scale)
        self.net_MGCN_nbhd = MGConv(self.hidden_dims, {'nbhd': self.L_nbhd}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_simi = MGConv(self.hidden_dims, {'simi': self.L_simi}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_cont = MGConv(self.hidden_dims, {'cont': self.L_cont}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        # self.net_MGCN_all = MGConv(self.hidden_dims, {'nbhd': self.L_nbhd, 'simi': self.L_simi, 'cont': self.L_cont},
        #                            2, self.num_vertices, self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_all  = MGConv(self.hidden_dims, {'all': self.L_nbhd}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_FC1 = tf.keras.layers.Dense(units=1, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))

    def __call__(self, x):

        # X.shape = [batch_size, time_length, num_vertices, feature_dims]
        def contxt_gate_rnn(inputs, net_CntxtGate, net_MGCN):
            gated_inputs = net_CntxtGate(inputs)
            temp_ext_out = self.net_tempExtractor(gated_inputs)  # [batch_size, num_vertices, hidden_dims] 
            network_out = net_MGCN(temp_ext_out)
            return network_out

        # output shape: [batch_size, num_vertices, ]
        network_out_nbhd = contxt_gate_rnn(x, self.net_CntxtGate_nbhd, self.net_MGCN_nbhd) 
        network_out_simi = contxt_gate_rnn(x, self.net_CntxtGate_simi, self.net_MGCN_simi)
        network_out_cont = contxt_gate_rnn(x, self.net_CntxtGate_cont, self.net_MGCN_cont)
        network_output = tf.concat([network_out_nbhd, network_out_simi, network_out_cont], axis=-1)
        # network_output = tf.math.maximum(network_out_nbhd, network_out_simi, network_out_cont)
        # network_output = self.net_MGCN_all(network_output)

        all_output = self.net_FC1(network_output)
        all_output = tf.reshape(all_output, shape=[self.batch_size, self.num_vertices])  # [batch_size, num_vertices]

        return all_output
    
    
class UNetSTMGCN(keras.Model):

    def __init__(self, batch_size, num_vertices, time_length, feature_dims, hidden_dims, en_ksize, de_ksize,
                 nbhd_adj, simi_adj, cont_adj, with_attention=True,
                 regularizer_scale=0.003, lambda_max=2):
        super(UNetSTMGCN, self).__init__()

        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.time_length = time_length
        self.feature_dims = feature_dims
        self.hidden_dims = hidden_dims
        self.regularizer_scale = regularizer_scale
        self.lambda_max = lambda_max
        self.with_attention = with_attention
        self.en_ksize = en_ksize
        self.de_ksize = de_ksize

        def get_sparse_laplacian(adj_mx):
            L_mx = calculate_scaled_laplacian(adj_mx, lambda_max=self.lambda_max)
            row, col, val = sp.find(L_mx)
            indices = np.column_stack((row, col))
            return tf.sparse.SparseTensor(indices, val, L_mx.shape)

        self.L_nbhd = get_sparse_laplacian(nbhd_adj)
        self.L_simi = get_sparse_laplacian(simi_adj)
        self.L_cont = get_sparse_laplacian(cont_adj)

        if self.with_attention:
            self.net_CntxtGate_nbhd = ContextGate(self.L_nbhd, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_simi = ContextGate(self.L_simi, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_cont = ContextGate(self.L_cont, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
        else:
            self.net_CntxtGate_nbhd = NoOpGate()
            self.net_CntxtGate_simi = NoOpGate()
            self.net_CntxtGate_cont = NoOpGate()

        self.net_tempExtractor = TempUNet(input_dims=self.feature_dims, num_vertices=self.num_vertices,
                                          time_length=self.time_length,
                                          mode='nearest', norm='batch', act_en='elu', act_de='leaky_relu',
                                          en_ksize=self.en_ksize, de_ksize=self.de_ksize,
                                          regularizer_scale=self.regularizer_scale, hidden_dims=self.hidden_dims)

        self.net_MGCN_nbhd = MGConv(self.hidden_dims, {'nbhd': self.L_nbhd}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_simi = MGConv(self.hidden_dims, {'simi': self.L_simi}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_cont = MGConv(self.hidden_dims, {'cont': self.L_cont}, 2, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        # self.net_MGCN_all = MGConv(self.hidden_dims, {'nbhd': self.L_nbhd, 'simi': self.L_simi, 'cont': self.L_cont},
        #                            2, self.num_vertices, self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_all = MGConv(self.hidden_dims, {'all': self.L_nbhd}, 2, self.num_vertices,
                                   self.regularizer_scale, lambda_max=lambda_max)
        self.net_FC1 = tf.keras.layers.Dense(units=1, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))

    def __call__(self, x):

        # X.shape = [batch_size, time_length, num_vertices, feature_dims]
        def contxt_gate_rnn(inputs, net_CntxtGate, net_MGCN):
            gated_inputs = net_CntxtGate(inputs)
            temp_ext_out = self.net_tempExtractor(gated_inputs)  # [batch_size, num_vertices, hidden_dims] 
            network_out = net_MGCN(temp_ext_out)
            return network_out

        # output shape: [batch_size, num_vertices, ]
        network_out_nbhd = contxt_gate_rnn(x, self.net_CntxtGate_nbhd, self.net_MGCN_nbhd)
        network_out_simi = contxt_gate_rnn(x, self.net_CntxtGate_simi, self.net_MGCN_simi)
        network_out_cont = contxt_gate_rnn(x, self.net_CntxtGate_cont, self.net_MGCN_cont)
        network_output = tf.concat([network_out_nbhd, network_out_simi, network_out_cont], axis=-1)
        # network_output = tf.math.maximum(network_out_nbhd, network_out_simi, network_out_cont)
        # network_output = self.net_MGCN_all(network_output)

        all_output = self.net_FC1(network_output)
        all_output = tf.reshape(all_output, shape=[self.batch_size, self.num_vertices])  # [batch_size, num_vertices]

        return all_output


class UNetSTMGCN2(keras.Model):

    def __init__(self, batch_size, num_vertices, time_length, feature_dims, hidden_dims, en_ksize, de_ksize,
                 nbhd_adj, simi_adj, cont_adj, max_diffusion_step=1, with_attention=True,
                 regularizer_scale=0.003, lambda_max=2):
        super(UNetSTMGCN2, self).__init__()

        self.batch_size = batch_size
        self.num_vertices = num_vertices
        self.time_length = time_length
        self.feature_dims = feature_dims
        self.hidden_dims = hidden_dims
        self.regularizer_scale = regularizer_scale
        self.lambda_max = lambda_max
        self.with_attention = with_attention
        self.en_ksize = en_ksize
        self.de_ksize = de_ksize
        self.gconv_max_diffusion_step = max_diffusion_step

        def get_sparse_laplacian(adj_mx):
            L_mx = calculate_scaled_laplacian(adj_mx, lambda_max=self.lambda_max)
            row, col, val = sp.find(L_mx)
            indices = np.column_stack((row, col))
            return tf.sparse.SparseTensor(indices, val, L_mx.shape)

        self.L_nbhd = get_sparse_laplacian(nbhd_adj)
        self.L_simi = get_sparse_laplacian(simi_adj)
        self.L_cont = get_sparse_laplacian(cont_adj)

        if self.with_attention:
            self.net_CntxtGate_nbhd = ContextGate(self.L_nbhd, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_simi = ContextGate(self.L_simi, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
            self.net_CntxtGate_cont = ContextGate(self.L_cont, self.batch_size, self.time_length, self.num_vertices,
                                                  self.feature_dims, self.regularizer_scale)
        else:
            self.net_CntxtGate_nbhd = NoOpGate()
            self.net_CntxtGate_simi = NoOpGate()
            self.net_CntxtGate_cont = NoOpGate()

        self.net_tempExtractor = TempUNet(input_dims=self.feature_dims, num_vertices=self.num_vertices,
                                          time_length=self.time_length,
                                          mode='nearest', norm='batch', act_en='elu', act_de='leaky_relu',
                                          en_ksize=self.en_ksize, de_ksize=self.de_ksize,
                                          regularizer_scale=self.regularizer_scale, hidden_dims=self.hidden_dims)

        self.net_MGCN_nbhd = MGConv(self.hidden_dims, {'nbhd': self.L_nbhd}, max_diffusion_step, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_simi = MGConv(self.hidden_dims, {'simi': self.L_simi}, max_diffusion_step, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        self.net_MGCN_cont = MGConv(self.hidden_dims, {'cont': self.L_cont}, max_diffusion_step, self.num_vertices,
                                    self.regularizer_scale, lambda_max=lambda_max)
        # self.net_MGCN_all = MGConv(self.hidden_dims, {'all': self.L_nbhd}, max_diffusion_step, self.num_vertices,
        #                            self.regularizer_scale, lambda_max=lambda_max)
        self.net_FC1 = tf.keras.layers.Dense(units=1, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))

    def __call__(self, x):

        # X.shape = [batch_size, time_length, num_vertices, feature_dims]
        def contxt_gate_rnn(inputs, net_CntxtGate, net_MGCN):
            gated_inputs = net_CntxtGate(inputs)
            temp_ext_out = self.net_tempExtractor(gated_inputs)  # [batch_size, num_vertices, hidden_dims] 
            network_out = net_MGCN(temp_ext_out)
            return network_out, temp_ext_out

        net_out_nbhd, temp_out_nbhd = contxt_gate_rnn(x, self.net_CntxtGate_nbhd, self.net_MGCN_nbhd)
        net_out_simi, _ = contxt_gate_rnn(x, self.net_CntxtGate_simi, self.net_MGCN_simi)
        net_out_cont, _ = contxt_gate_rnn(x, self.net_CntxtGate_cont, self.net_MGCN_cont)
        network_output = tf.concat([net_out_nbhd, net_out_simi, net_out_cont, temp_out_nbhd], axis=-1)
        # network_output = tf.math.maximum(network_out_nbhd, network_out_simi, network_out_cont)
        # network_output = self.net_MGCN_all(network_output)

        all_output = self.net_FC1(network_output)
        all_output = tf.reshape(all_output, shape=[self.batch_size, self.num_vertices])  # [batch_size, num_vertices]

        return all_output
