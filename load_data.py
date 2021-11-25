import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib
import joblib
# from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
scaler = joblib.load('scaler.pkl')

def generate(seq, time_steps, num_vertices, feature_dims):
    """
    Generate the data from the given sequence
    :param seq: the sequence for generation
    :param time_steps:
    :param num_vertices:
    :param feature_dims:
    :return: list of X, Y
    """
    # X = []
    # y = []
    # for i in range(seq.shape[0] - time_steps):
    #     X.append([seq[i:i + time_steps]])
    #     y.append([seq[i + time_steps]])

    # X = np.array(X, dtype=np.float32)
    # X = X.reshape((-1, time_steps, num_vertices, feature_dims))

    # Y = np.array(y, dtype=np.float32)
    # Y = Y[:,:,:,0]
    # Y = Y.reshape((-1, num_vertices))
    X = []
    y = []
    for i in range(seq.shape[0] - time_steps):
        X.append([seq[i:i + time_steps]])
        y.append([seq[i + time_steps]])
        
    X = np.array(X, dtype=np.float32)
    X = X.transpose(0, 2, 3, 1)
    
    Y = np.array(y, dtype=np.float32)
    Y = np.squeeze(Y)
    return X, Y


# def load_data(path_data, time_steps, num_vertices, feature_dims, nrows, normalize_type='vertex'):
def load_data(path_data, time_steps, num_vertices, feature_dims, nrows=None, skiprows=0, normalize_type='vertex'):
    # load dataset
    
    # dataset.shape = [time_length, num_vertices, feature_dims]
    if path_data.endswith('.csv'):
        dataset = pd.read_csv(path_data, index_col=0, nrows=nrows, skiprows=skiprows)
        dataset = dataset.values.astype('float32')
    else:
        dataset = np.load(path_data, allow_pickle=True)['arr_0']
        dataset = dataset[:nrows,:,:feature_dims]
        dataset = dataset.astype('float32')
    
    print('dataset.shape = {} -------------------------------------------------------------------'.format(dataset.shape))
    
    # normalize features
    if normalize_type == 'all':
        if feature_dims == 6:
            scaled = scaler.transform(dataset[:,:,-1].reshape(-1, 1))
            scaled = scaled.reshape(dataset[:,:,-1].shape)
            dataset[:,:,-1] = scaled
        
        scaled = scaler.transform(dataset[:,:,0].reshape(-1, 1))
        scaled = scaled.reshape(dataset[:,:,0].shape)
        dataset[:,:,0] = scaled
        
        scaled = dataset
    else:
        if feature_dims == 6:
            scaled = scaler.transform(dataset[:,:,-1])
            dataset[:,:,-1] = scaled
        # scaled = scaler.fit_transform(dataset[:,:,0])
        # dataset[:,:,0] = scaled
        # scaled = dataset
        scaled = scaler.transform(dataset)
            
    # split into train and test sets
    n_train = int(len(scaled) * 0.8) + 1
    n_val = int(len(scaled) * 0.2)
    # _test = int(len(scaled) * 0.2)

    train = scaled[:n_train, :]
    val = scaled[n_train:n_train + n_val, :]
    # test = scaled[n_train + n_val:, :]
        
    X_train, y_train = generate(train, time_steps, num_vertices, feature_dims)
    X_val, y_val = generate(val, time_steps, num_vertices, feature_dims)
    # X_test, y_test = generate(test, time_steps, num_vertices, feature_dims)
    
    return X_train, y_train, X_val, y_val #, X_test, y_test


def load_raw(path_data, time_steps, feature_dims, nrows=None, skiprows=0, normalize_type='vertex'):

    # dataset.shape = [time_length, num_vertices, feature_dims]
    if path_data.endswith('.csv'):
        dataset = pd.read_csv(path_data, index_col=0, nrows=nrows, skiprows=skiprows)
        dataset = dataset.values.astype('float32')
    else:
        dataset = np.load(path_data, allow_pickle=True)['arr_0']
        dataset = dataset[:nrows, :, :feature_dims]
        dataset = dataset.astype('float32')
    
    # normalize features
    if normalize_type == 'all':
        if feature_dims == 6:
            scaled = scaler.transform(dataset[:, :, -1].reshape(-1, 1))
            scaled = scaled.reshape(dataset[:, :, -1].shape)
            dataset[:, :, -1] = scaled

        scaled = scaler.transform(dataset[:, :, 0].reshape(-1, 1))
        scaled = scaled.reshape(dataset[:, :, 0].shape)
        dataset[:, :, 0] = scaled
        scaled = dataset
    else:
        if feature_dims == 6:
            scaled = scaler.transform(dataset[:, :, -1])
            dataset[:, :, -1] = scaled
        scaled = scaler.transform(dataset)

    # split into train and test sets
    n_train = int(len(scaled) * 0.8) + 1
    # n_val = int(len(scaled) * 0.2)
    
    train = tf.convert_to_tensor(scaled[:n_train, :], dtype=tf.float32)
    # val   = tf.convert_to_tensor(scaled[n_train:,:], dtype=tf.float32)
    val   = tf.convert_to_tensor(scaled[n_train - time_steps*96*7:,:], dtype=tf.float32)
    
    # train_idx = list(range(train.shape[0] - time_steps))
    # val_idx = list(range(val.shape[0] - time_steps))
    train_len = train.shape[0] - time_steps
    val_len   = val.shape[0] - time_steps

    return train, val, train_len, val_len


def load_adj_npz(npz_nbhd_path, npz_simi_path, npz_cont_path):
    """
    Load adjacent matrix from a npz file which comprising 3 sparse csr matrix
    :param npz_nbhd_path:
    :param npz_nbhd_path:
    :param npz_nbhd_path:
    :return:
    """
    
    nbhd_adj = scipy.sparse.load_npz(npz_nbhd_path)
    simi_adj = scipy.sparse.load_npz(npz_simi_path)
    cont_adj = scipy.sparse.load_npz(npz_cont_path)

    return nbhd_adj, simi_adj, cont_adj


def load_adj_csv(nbhd_path, simi_path, cont_path, output_sparse=False):
    """
    Load adjacent matrix from csv file
    Notice that csv is represented by the dense matrix expression
    :param nbhd_path:
    :param simi_path:
    :param cont_path:
    :return:
    """
    
    # nbhd_adj = pd.read_csv(nbhd_path, header=None, index_col=None)
    nbhd_adj = pd.read_csv(nbhd_path, index_col=0)
    nbhd_adj = sparse.csr_matrix(nbhd_adj.values) if output_sparse else np.mat(nbhd_adj)

    # simi_adj = pd.read_csv(simi_path, header=None, index_col=None)
    simi_adj = pd.read_csv(simi_path, index_col=0)
    simi_adj = sparse.csr_matrix(simi_adj.values) if output_sparse else np.mat(simi_adj)

    # cont_adj = pd.read_csv(cont_path, header=None, index_col=None)
    cont_adj = pd.read_csv(cont_path, index_col=0)
    cont_adj = sparse.csr_matrix(cont_adj.values) if output_sparse else np.mat(cont_adj)

    return nbhd_adj, simi_adj, cont_adj


def get_batch(X, y, batch_size):
    idx = np.random.randint(X.shape[0] - batch_size)
    x_batch = X[idx: idx + batch_size]
    y_batch = y[idx: idx + batch_size]
    return x_batch, y_batch


def get_batch_from_raw(data, batch_size, time_steps):
    indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=data.shape[0] - time_steps, dtype=tf.int64)
    x_batch = tf.map_fn(lambda x: data[x:x+time_steps], indices, dtype=tf.float32)
    x_batch = tf.expand_dims(x_batch, axis=-1)
    y_batch = tf.map_fn(lambda x: data[x+time_steps], indices, dtype=tf.float32)
    return x_batch, y_batch


def generate3feat(seq, time_steps, num_vertices, feature_dims):
    """
    Generate the data from the given sequence
    :param seq: the sequence for generation
    :param time_steps:
    :param num_vertices:
    :param feature_dims:
    :return: list of X, Y
    """
    X_Q    = []
    X_day  = []
    X_week = []
    y = []
    for i in range(seq.shape[0] - time_steps*96*7):
        X_Q.append([seq[i + time_steps*96*7 - time_steps:i + time_steps*96*7]])
        X_day.append([seq[[n for n in range(i+time_steps*96*7 - 16*96, i+time_steps*96*7, 96)]]])
        X_week.append([seq[[n for n in range(i, i+time_steps*96*7, 96*7)]]])
        y.append([seq[i + time_steps*96*7]])
        
    X_Q = np.array(X_Q, dtype=np.float32)
    X_Q = X_Q.transpose(0, 2, 3, 1)
    
    X_day = np.array(X_day, dtype=np.float32)
    X_day = X_day.transpose(0, 2, 3, 1)
    
    X_week = np.array(X_week, dtype=np.float32)
    X_week = X_week.transpose(0, 2, 3, 1)
    
    X = np.concatenate([X_Q, X_day, X_week], axis=3)
    
    Y = np.array(y, dtype=np.float32)
    Y = np.squeeze(Y)
    return X, Y


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
    
    # split into train and test sets
    n_train = int((len(scaled)-time_steps*96*7) * 0.75) + 1
    
    train = scaled[:n_train+time_steps*96*7, :]
    val   = scaled[n_train:,:]
        
    X_train, y_train = generate3feat(train, time_steps, num_vertices, feature_dims)
    X_val,   y_val   = generate3feat(val, time_steps, num_vertices, feature_dims)
    
    return X_train, y_train, X_val, y_val