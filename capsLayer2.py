import numpy as np
import tensorflow as tf
epsilon = 1e-9

class CapsLayer(object):

    def __init__(self, num_outputs_firstCapsLayer, num_outputs_secondCapsLayer, vec_len_firstCapsLayer, vec_len_secondCapsLayer,
                 batch_size, iter_routing, embedding_size, layer_type='NextCapsule'):
        self.num_outputs_firstCapsLayer = num_outputs_firstCapsLayer
        self.num_outputs_secondCapsLayer = num_outputs_secondCapsLayer
        self.vec_len_firstCapsLayer = vec_len_firstCapsLayer
        self.vec_len_secondCapsLayer = vec_len_secondCapsLayer
        self.layer_type = layer_type
        self.batch_size = batch_size
        self.iter_routing = iter_routing
        self.embedding_size = embedding_size


    def __call__(self, input):

        if self.layer_type == 'FirstCapsule':
            capsules = squash(input)
            return(capsules)

        if self.layer_type == 'NextCapsule':

            # Reshape the input into [batch_size, num_outputs_firstCapsLayer=9, 1, vec_len_firstCapsLayer, 1]
            self.input = tf.reshape(input, shape=(-1, input.shape[1].value,
                                                  1, input.shape[-2].value, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                b_IJ = tf.constant(np.zeros([self.batch_size, input.shape[1].value, self.num_outputs_secondCapsLayer, 1, 1], dtype=np.float32))
                capsules = routing(self.input, b_IJ, batch_size=self.batch_size, iter_routing=self.iter_routing,
                                   num_caps_i=self.num_outputs_firstCapsLayer, num_caps_j=self.num_outputs_secondCapsLayer,
                                   len_u_i=self.vec_len_firstCapsLayer, len_v_j=self.embedding_size)
                capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


def routing(input, b_IJ, batch_size, iter_routing, num_caps_i, num_caps_j, len_u_i, len_v_j):
    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, num_caps_i, num_caps_j, len_u_i, len_v_j), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01, seed=1234))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    input = tf.tile(input, [1, 1, num_caps_j, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])

    # in last 2 dims:
    u_hat = tf.matmul(W, input, transpose_a=True)
    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            c_IJ = tf.nn.softmax(b_IJ, axis=1) * num_caps_i #axis=1 # original code

            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, num_caps_j, len_v_j, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)

            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                v_J = squash(s_J)
                # line 7:
                v_J_tiled = tf.tile(v_J, [1, num_caps_i, 1, 1, 1])
                b_IJ = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

