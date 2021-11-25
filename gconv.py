import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_normalized_laplacian(adj_mx):
    """
    # L = D^{-1/2} (D-A) D^{-1/2} = I - D^{-1/2} * A * D^{-1/2}
    # D = diag(A 1)
    :param adj:
    :return:
    """
    if not sp.issparse(adj_mx):
        adj_mx = sp.csr_matrix(adj_mx)

    # compute D^{-1/2}
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt * adj_mx * d_mat_inv_sqrt


def calculate_laplacian(adj_mx, lambda_max=2, undirected=True):
    if not sp.issparse(adj_mx):
        adj_mx = sp.csr_matrix(adj_mx)

    if undirected:
        adj_mx = 0.5 * (adj_mx + adj_mx.T)

    L = calculate_normalized_laplacian(adj_mx)

    return L.astype(np.float32)


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if not sp.issparse(adj_mx):
        adj_mx = sp.csr_matrix(adj_mx)

    if undirected:
        adj_mx = 0.5 * (adj_mx + adj_mx.T)

    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]

    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def calculate_random_walk_matrix(adj_mx):
    if not sp.issparse(adj_mx):
        adj_mx = sp.csr_matrix(adj_mx)

    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


class MGConv(tf.keras.Model):
    def __init__(self,
                 output_dims,
                 L_dict,
                 max_diffusion_step,
                 num_vertices,
                 regularizer_scale,
                 num_proj=None,
                 activation=tf.nn.tanh,
                 lambda_max=None,
                 filter_type="laplacian"):
        """
        Initializer of the graph convolution layer
        :param output_dims:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_vertices:
        :param num_proj:
        :param activation:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        """
        
        super(MGConv, self).__init__()
        self._activation = activation
        self._num_vertices = num_vertices
        self._num_proj = num_proj
        self._output_dims = output_dims
        self._max_diffusion_step = max_diffusion_step
        self.regularizer_scale = regularizer_scale
        self.lambda_max = lambda_max

        self.net_FC = tf.keras.layers.Dense(units=self._output_dims, activation='elu',
                                            kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_scale))
        self.laplacians = {}

        for L_key in L_dict:
            L_mx = L_dict[L_key]
            if filter_type == "laplacian":
                self.laplacians[L_key] = tf.sparse.reorder(L_mx)
    
    @property
    def output_size(self):
        output_size = self._num_vertices * self._output_dims
        if self._num_proj is not None:
            output_size = self._num_vertices * self._num_proj
        return output_size

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def __call__(self, inputs):
        # Reshape input to (batch_size, num_vertices, input_dim)
        batch_size = inputs.get_shape()[0]
        inputs = tf.reshape(inputs, [batch_size, self._num_vertices, -1])
        input_size = inputs.get_shape()[2]
        x = inputs
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_vertices, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_vertices, input_size * batch_size])
        x_dict = {}

        if self._max_diffusion_step == 0:
            pass
        else:
            for l_key in self.laplacians:
                x = tf.expand_dims(x0, axis=0)
                L_mx = self.laplacians[l_key]
                x1 = tf.sparse.sparse_dense_matmul(L_mx, x0)
                x = self._concat(x, x1)
                for _ in range(2, self._max_diffusion_step + 1):
                    # Chebyshev polynomial
                    x2 = 2 * tf.sparse.sparse_dense_matmul(L_mx, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

                x_dict[l_key] = x

        for l_key in x_dict:
            x = x_dict[l_key]
            num_matrices = 1 * self._max_diffusion_step + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_vertices, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_vertices, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_vertices, input_size * num_matrices])
            x = self.net_FC(x)
            x_dict[l_key] = tf.reshape(x, [batch_size, self._num_vertices, self._output_dims])

        # Reshape res back to: (batch_size, num_node, state_dim)
        return tf.concat(list(x_dict.values()), axis=-1)
