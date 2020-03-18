#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from capsNet2 import Caps2NE
import pickle as cPickle
from utils import *

np.random.seed(1234)
tf.set_random_seed(1234)

# Parameters
# ==================================================
parser = ArgumentParser("Caps2NE", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--data", default="./graph/", help="Data sources.")
parser.add_argument("--run_folder", default="../", help="Data sources.")
parser.add_argument("--name", default="cora.128.10.trans.pickle", help="Name of the dataset.")
parser.add_argument("--embedding_dim", default=8, type=int, help="Dimensionality of character embedding")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=8, type=int, help="Batch Size")
parser.add_argument("--idx_time", default=1, type=int, help="")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--saveStep", default=1, type=int, help="")
parser.add_argument("--allow_soft_placement", default=True, type=bool, help="Allow device soft device placement")
parser.add_argument("--log_device_placement", default=False, type=bool, help="Log placement of ops on devices")
parser.add_argument("--model_name", default='cora_trans', help="")
parser.add_argument("--useInductive", action='store_true')
parser.add_argument('--num_sampled', default=256, type=int, help='')
parser.add_argument('--iter_routing', default=1, type=int, help='number of iterations in routing algorithm')
parser.add_argument('--num_outputs_secondCapsLayer', default=1, type=int, help='')
parser.add_argument("--is_trainable", default=False, type=bool, help="")
parser.add_argument("--write_file", default='cora', help="")

args = parser.parse_args()
print(args)


class Batch_Loader_RW(object):
    def __init__(self, walks, batch_size=10):
        self.lstArr = range(10)  # depends on the average number of edges per node, for POS PPI BlogCatalog
        if str(args.name).split('.')[0] in ['cora', 'citeseer', 'pubmed']:
            self.lstArr = [3, 4, 5, 6]  # just ~ 2 edges per node
        self.walks = walks
        self.batch_size = batch_size
        self.data_size = len(self.walks)
        self.sequence_length = np.shape(self.walks)[1]
        self.check()

    def __call__(self):
        idxs = np.random.randint(0, self.data_size, self.batch_size)
        return self.generatedata(self.walks[idxs])

    def generatedata(self, _input):
        arrX = []
        arrY = []
        for tmp in _input:
            for i in self.lstArr:
                arrX.append([tmp[j] for j in range(0, self.sequence_length) if j != i])
                arrY.append([tmp[i]])
        return np.array(arrX), np.array(arrY)

    def check(self):
        _dict = set()
        for walk in self.walks:
            for tmp in walk:
                if tmp not in _dict:
                    _dict.add(int(tmp))
        self._dict = _dict


# Load data
print("Loading data...")

with open(args.data + args.name, 'rb') as f:
    walks = cPickle.load(f)
batch_rw = Batch_Loader_RW(walks, args.batch_size)

features_matrix = []  # for POS, PPI, blogcatalog
vec_len_firstCapsLayer = args.embedding_dim  # can be defined by another hyper-parameter, just fixed = embedding size for simplification
vocab_size = len(batch_rw._dict)

if str(args.name).split('.')[0] in ['cora', 'citeseer', 'pubmed']:  # for cora, pubmed, citeseer
    features, _ = load_data(str(args.name).split('.')[0])
    features_matrix, spars = preprocess_features(features)
    features_matrix = np.array(features_matrix, dtype=np.float32)
    vec_len_firstCapsLayer = features_matrix.shape[1]
    vocab_size = features_matrix.shape[0]

print("Loading data... finished!")

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=args.allow_soft_placement,
                                  log_device_placement=args.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        capsNet = Caps2NE(sequence_length=batch_rw.sequence_length - 1,
                          embedding_size=args.embedding_dim,
                          vocab_size=vocab_size,
                          iter_routing=args.iter_routing,
                          batch_size=len(batch_rw.lstArr) * args.batch_size,
                          num_sampled=args.num_sampled,
                          initialization=features_matrix,
                          vec_len_firstCapsLayer=vec_len_firstCapsLayer
                          )

        # Define Training procedure
        # optimizer = tf.contrib.opt.NadamOptimizer(1e-3)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(capsNet.total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(args.run_folder, "runs_Caps2NE_trans", args.model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        graph = tf.get_default_graph()


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                capsNet.input_x: x_batch,
                capsNet.input_y: y_batch
            }
            _, step, loss = sess.run([train_op, global_step, capsNet.total_loss], feed_dict)
            return loss


        num_batches_per_epoch = int((batch_rw.data_size - 1) / args.batch_size) + 1
        for epoch in range(1, args.num_epochs + 1):
            loss = 0
            for batch_num in range(num_batches_per_epoch):
                x_batch, y_batch = batch_rw()
                loss += train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
            # print(loss)
            if epoch % args.saveStep == 0:
                # It will give tensor object
                embeddingW = graph.get_tensor_by_name('W:0')
                # To get the value (numpy array)
                embeddingW_value = sess.run(embeddingW)
                with open(checkpoint_prefix + '-' + str(epoch), 'wb') as f:
                    cPickle.dump(embeddingW_value, f)
                print("Save embeddings to {}\n".format(checkpoint_prefix + '-' + str(epoch)))

                # input_feature = graph.get_tensor_by_name('input_feature/input_feature_2:0')
