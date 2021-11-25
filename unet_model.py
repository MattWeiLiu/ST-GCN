# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras


def get_norm(name):
    return keras.layers.BatchNormalization() if name == 'batch' else None


def get_activation(name):
    if name == 'relu':
        activation = keras.layers.ReLU()
    elif name == 'elu':
        activation = keras.layers.ELU()
    elif name == 'leaky_relu':
        activation = keras.layers.LeakyReLU(alpha=0.2)
    elif name == 'tanh':
        activation = keras.layers.Activation('tanh')
    elif name == 'sigmoid':
        activation = keras.layers.Activation('sigmoid')
    else:
        activation = None
    return activation


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid'):
        super(Conv1DTranspose, self).__init__()
        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(filters, (kernel_size, 1), (strides, 1), padding)

    def __call__(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2d_transpose(x)
        x = tf.squeeze(x, axis=2)
        return x


class ConvTranspose1DSame(keras.Model):

    def __init__(self, out_channels, kernel_size, stride):
        super(ConvTranspose1DSame, self).__init__()

        padding, output_padding = self.deconv_same_pad(kernel_size, stride)

        self.trans_conv = Conv1DTranspose(filters=out_channels, kernel_size=kernel_size, strides=stride,
                                          padding=padding)

    @staticmethod
    def deconv_same_pad(ksize, stride):
        pad = (ksize - stride + 1) // 2
        outpad = 2 * pad + stride - ksize
        return pad, outpad

    def __call__(self, x):
        return self.trans_conv(x)


class UpBlock(keras.Model):

    def __init__(self, mode='nearest', scale=2, channel=None, kernel_size=4):
        super(UpBlock, self).__init__()

        self.mode = mode
        layers = []
        if mode == 'deconv':
            layers.append(ConvTranspose1DSame(channel, kernel_size, stride=scale))
        else:
            layers.append(keras.layers.UpSampling1D(size=scale))
        self.up = keras.Sequential(layers)

    def __call__(self, x):
        return self.up(x)


class EncodeBlock(keras.Model):

    def __init__(self, in_channels, out_channels, kernel_size, stride, normalization=None, activation=None):
        super(EncodeBlock, self).__init__()

        self.input_dims = in_channels
        self.c_out = out_channels

        layers = [keras.layers.Conv1D(filters=self.c_out, kernel_size=kernel_size, strides=stride, padding="same")]
        if normalization:
            layers.append(get_norm(normalization))
        if activation:
            layers.append(get_activation(activation))
        self.encode = keras.Sequential(layers)

    def __call__(self, x):
        return self.encode(x)


class DecodeBlock(keras.Model):

    def __init__(self, c_from_up, c_from_down, c_out,
                 mode='nearest', kernel_size=4, scale=2, normalization='batch', activation='relu'):
        super(DecodeBlock, self).__init__()

        self.c_from_up = c_from_up
        self.c_from_down = c_from_down
        self.input_dims = c_from_up + c_from_down
        self.c_out = c_out

        self.up = UpBlock(mode, scale, channel=c_from_up, kernel_size=scale)

        layers = [keras.layers.Conv1D(filters=self.c_out, kernel_size=kernel_size, strides=1, padding="same")]
        if normalization:
            layers.append(get_norm(normalization))
        if activation:
            layers.append(get_activation(activation))

        self.decode = keras.Sequential(layers)

    def __call__(self, x, concat=None):
        out = self.up(x)
        if self.c_from_down > 0:
            # print('[{} , {}]'.format(out.shape, concat.shape))
            out = tf.concat([out, concat], axis=-1)   # TensorFlow is channel-last
        out = self.decode(out)
        return out


class TempUNet(keras.Model):

    def __init__(self, input_dims, num_vertices, time_length,
                 mode='nearest', norm='batch', act_en='elu', act_de='leaky_relu',
                 en_ksize=[7, 5, 5, 3, 3, 3, 3], de_ksize=[3] * 7,
                 regularizer_scale=0.0003, hidden_dims=64):
        super(TempUNet, self).__init__()

        self.num_vertices = num_vertices
        self.time_length = time_length
        self.n_en = len(en_ksize)
        self.n_de = len(de_ksize)
        self.regularizer_scale = regularizer_scale
        self.hidden_dims = hidden_dims

        assert self.n_en == self.n_de, 'The number layer of Encoder and Decoder must be equal.'
        assert self.n_en >= 1, 'The number layer of Encoder and Decoder must be greater than 1.'

        self.en = []
        self.en.append(EncodeBlock(input_dims, 4, en_ksize[0], 2, None, None))
        for k_en in en_ksize[1:]:
            input_dims = self.en[-1].c_out
            c_out = min(input_dims * 2, 32)
            self.en.append(EncodeBlock(input_dims, c_out, k_en, stride=2, normalization=norm, activation=act_en))

        self.de = []
        for i, k_de in enumerate(de_ksize):

            c_from_up = self.en[-1].c_out if i == 0 else self.de[-1].c_out
            c_out = c_from_down = self.en[-i - 1].input_dims

            self.de.append(DecodeBlock(c_from_up, c_from_down, c_out, mode, k_de,
                                       scale=2, normalization=norm, activation=act_de))
        
        self.net_Flatten = tf.keras.layers.Flatten()
        self.net_Dropout = keras.layers.Dropout(0.1)
        self.net_FC = tf.keras.layers.Dense(units=hidden_dims, activation='tanh',
                                            kernel_regularizer=tf.keras.regularizers.l2(regularizer_scale))

    def __call__(self, x):

        batch_size, time_length, num_vertices, feature_dims = x.get_shape()
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch_size, num_vertices, time_length, feature_dims]
        x = tf.reshape(x, [-1, time_length, feature_dims])  # [batch_size * num_vertices, time_length, hidden_dims]

        x_en = [x]
        for encode in self.en:
            x = encode(x)
            x_en.append(x)

        embeds = []
        for i, decode in enumerate(self.de):
            x = decode(x, x_en[-i - 2])
            flat_x = self.net_Flatten(x)
            embeds.append(flat_x)

        embeds = tf.concat(embeds, axis=-1)
        embeds = self.net_Dropout(embeds)
        embeds = self.net_FC(embeds)

        embeds = tf.reshape(embeds, [batch_size, num_vertices, self.hidden_dims])

        return embeds
