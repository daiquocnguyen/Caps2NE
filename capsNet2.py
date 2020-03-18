import tensorflow as tf

from capsLayer2 import CapsLayer
import math

class Caps2NE(object):
    def __init__(self, sequence_length, embedding_size, vocab_size, iter_routing, vec_len_firstCapsLayer,
                 initialization=[], batch_size=256, num_sampled=256):
        # Placeholders for input, output
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [batch_size, 1], name="input_y")
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.iter_routing = iter_routing
        self.num_outputs_firstCapsLayer = sequence_length
        self.vec_len_firstCapsLayer = vec_len_firstCapsLayer
        self.num_outputs_secondCapsLayer = 1
        self.vec_len_secondCapsLayer = embedding_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        # Embedding layer
        with tf.name_scope("input_feature"):
            if initialization != []:
                self.input_feature = tf.get_variable(name="input_feature_1", initializer=initialization, trainable=False)
            else:
                self.input_feature = tf.Variable(
                    tf.random_uniform([vocab_size, vec_len_firstCapsLayer], -math.sqrt(1.0 / vec_len_firstCapsLayer),
                                      math.sqrt(1.0 / vec_len_firstCapsLayer), seed=1234), name="input_feature_2")

        self.embedded_chars = tf.nn.embedding_lookup(self.input_feature, self.input_x)
        self.X = tf.expand_dims(self.embedded_chars, -1)

        self.build_arch()
        self.loss()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)

        tf.logging.info('Seting up the main structure')

    def build_arch(self):

        # The first capsule layer
        with tf.variable_scope('FirstCaps_layer'):
            self.primaryCaps = CapsLayer(num_outputs_firstCapsLayer=self.num_outputs_firstCapsLayer,
                                    num_outputs_secondCapsLayer=self.num_outputs_secondCapsLayer,
                                    vec_len_firstCapsLayer=self.vec_len_firstCapsLayer, vec_len_secondCapsLayer=self.vec_len_secondCapsLayer,
                                    layer_type='FirstCapsule', embedding_size=self.embedding_size,
                                    batch_size=self.batch_size, iter_routing=self.iter_routing)
            self.caps1 = self.primaryCaps(self.X)

            # assert caps1.get_shape() == [self.batch_size, num_outputs_firstCapsLayer, vec_len_firstCapsLayer, 1]

        # The second capsule layer
        with tf.variable_scope('SecondCaps_layer'):
            self.digitCaps = CapsLayer(num_outputs_firstCapsLayer=self.num_outputs_firstCapsLayer,
                                  num_outputs_secondCapsLayer=self.num_outputs_secondCapsLayer,
                                  vec_len_firstCapsLayer=self.vec_len_firstCapsLayer, vec_len_secondCapsLayer=self.vec_len_secondCapsLayer,
                                  layer_type='NextCapsule', embedding_size=self.embedding_size,
                                  batch_size=self.batch_size, iter_routing=self.iter_routing)
            self.caps2 = self.digitCaps(self.caps1)

    def loss(self):
        self.caps2reshape = tf.reshape(self.caps2, (self.batch_size, self.embedding_size))

        with tf.name_scope("embedding"):
            self.embedding_matrix = tf.get_variable(
                    "W", shape=[self.vocab_size, self.embedding_size],
                    initializer=tf.contrib.layers.xavier_initializer(seed=1234))

            self.softmax_biases = tf.Variable(tf.zeros([self.vocab_size]))

        self.total_loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.embedding_matrix, biases=self.softmax_biases, inputs=self.caps2reshape,
                                       labels=self.input_y, num_sampled=self.num_sampled, num_classes=self.vocab_size))

