import os
import numpy as np
import tensorflow as tf
import time
import math

import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Clock():
    def __init__(self, epochs):
        self.clock = 0
        self.clocks = dict()
        self.limits = dict()
        self.status = dict()
        self.epochs = epochs
        for epoch in self.epochs:
            self.clocks[epoch] = 0
            self.limits[epoch] = epoch
            self.status[epoch] = True

    def update(self):
        for epoch in self.epochs:
            self.clocks[epoch] += 1
            if self.limits[epoch] == self.clocks[epoch]:
                self.status[epoch] = True
                self.clocks[epoch] = 0

    def check(self, epoch):
        if self.status[epoch] == True:
            self.status[epoch] = False
            return True

class Model(object):
    def __init__(self,
                 embedding_size, # embedding dim for both input/output
                 learn_rate, # initial learning rate for gradient descent
                 i_index_size, # number of nodes in input space
                 j_index_size, # number of nodes in output space
                 c=1
                ):

        self.embedding_size = embedding_size
        self.init_range = 1/embedding_size
        self.learn_rate = learn_rate
        self.i_index_size = i_index_size
        self.j_index_size = j_index_size
        self.indexJ = None
        self.indexI = None
        self.wi = None
        self.wj = None
        self.bi = None
        self.bj = None

        self.indexI = tf.placeholder(tf.int32, shape=[None], name="indexI")
        self.IW = tf.Variable(
            tf.random_uniform([self.i_index_size, self.embedding_size], -self.init_range, self.init_range), name="IW")
        self.IB = tf.Variable(tf.random_uniform([self.i_index_size], -self.init_range, self.init_range), name="IB")
        self.wi = tf.nn.embedding_lookup(self.IW, self.indexI, name="wi")
        self.bi = tf.nn.embedding_lookup(self.IB, self.indexI, name="bi")

        self.indexJ = tf.placeholder(tf.int32, shape=[None], name="indexJ")
        self.JW = tf.Variable(
            tf.random_uniform([self.j_index_size, self.embedding_size], -self.init_range, self.init_range), name="JW")
        self.JB = tf.Variable(tf.random_uniform([self.j_index_size], -self.init_range, self.init_range), name="JB")
        self.wj = tf.nn.embedding_lookup(self.JW, self.indexJ, name="wj")
        self.bj = tf.nn.embedding_lookup(self.JB, self.indexJ, name="bj")

        self.Xij = tf.placeholder(tf.float32, shape=[None], name="Xij")

        wiwjProduct = tf.reduce_sum(tf.multiply(self.wi,self.wj), 1)
        logXij = tf.log(c+self.Xij)
        dist = tf.square(tf.add_n([wiwjProduct, self.bi, self.bj, tf.negative(logXij)]))
        self.loss = tf.reduce_sum(dist, name="loss")

        tf.summary.scalar("loss", self.loss)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learnRate = tf.Variable(learn_rate, trainable=False, name="learnRate")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learnRate, name="optimizer").minimize(
                self.loss, global_step=self.global_step)


class gvnr:
    def __init__(self, saver = None, callback = None):
        self.saver = saver
        self.callback = callback
        self.X = None # nodes cooccurrences matrix
        self.J_matrix = None
        self.I_matrix = None

    def chuncker(self, iterable, n):
        """
        grouper([ABCDEFG], 3) --> [[ABC],[DEF],[G]]
        """
        ind = range(0, len(iterable), n)
        for i in range(len(ind) - 1):
            yield iterable[ind[i]:ind[i + 1]]
        if ind[-1] < len(iterable):
            yield iterable[ind[-1]:len(iterable)]

    def generate_batches(self):
        """
        occurrences_counts = self.X.sum(axis=1)
        occurrences_counts[occurrences_counts == 0] = 1
        flatten_probs = np.power(1 / occurrences_counts, 0.75)
        nodes_negsampling_prob = np.squeeze(np.asarray(flatten_probs / flatten_probs.sum()))
        """
        data = np.array(self.X.data, dtype=np.float32)
        indices = self.X.nonzero()
        cols = np.array(indices[1], dtype=np.int32)
        rows = np.array(indices[0], dtype=np.int32)

        M = len(cols)
        cols_neg = np.tile(cols, self.k_neg)
        rows_neg = np.random.randint(0, self.X.shape[0], self.k_neg * M)
        #rows_neg = np.random.choice(np.arange(self.X.shape[0]), size=self.k_neg*M, p=nodes_negsampling_prob)

        cols = np.hstack((cols,cols_neg))
        rows = np.hstack((rows, rows_neg))
        data = np.hstack((data, np.zeros(self.k_neg*M)))

        ind = np.arange(len(data))
        np.random.shuffle(ind)
        data = data[ind]
        cols = cols[ind]
        rows = rows[ind]
        logger.debug("Shape of X=%s", self.X.shape)
        for ind in self.chuncker(range(0, len(data)), self.batch_size):
            yield rows[ind], cols[ind], data[ind]


    def fit(self, X, embedding_size = 80, batch_size = 128, n_epochs = 2, learn_rate=0.001, k_neg = 1, x_min=1, c=1):
        self.X = X
        self.x_min = x_min
        logger.debug("Size of X.data before filtering:%s", len(self.X.data))
        self.X.data[self.X.data <= self.x_min] = 0
        self.X.eliminate_zeros()
        logger.debug("Size of X.data after filtering:%s", len(self.X.data))
        logger.debug(" ".join([str(v) for v in ["Shape of cooccurrences matrix : ", X.shape]]))
        logger.debug(" ".join([str(v) for v in ["Density of cooccurrences matrix: ", " (", len(X.data) * 100 / (X.shape[0] * X.shape[1]), "%)"]]))
        logger.debug(
            "X => Min=%s/Max=%s/Mean=%s/Std=%s/Sum=%s:",
            X.data.min(),
            X.data.max(),
            X.data.mean(),
            X.data.std(),
            X.data.sum()
        )
        self.k_neg = k_neg
        self.links_count = len(X.data)
        self.i_index_size = self.X.shape[0]
        self.j_index_size = self.X.shape[1]
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.embedding_size = embedding_size
        self.num_batches = math.ceil(self.links_count*(1+self.k_neg) / self.batch_size) * n_epochs
        self.learn_rate = learn_rate

        logger.debug("Number of data={0}, number of batches={1}, number of epochs={2}".format(
                    self.links_count, self.num_batches, n_epochs))

        loss = 0
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                model = Model(
                      embedding_size = self.embedding_size,
                      learn_rate = self.learn_rate,
                      i_index_size = self.i_index_size,
                      j_index_size = self.j_index_size,
                      c=1)
                lr = 0
                init_op = tf.global_variables_initializer()
                self.session.run(init_op)
                start = time.time()
                clock = Clock([10, 100, 1000, 10000])
                saver_step = 0.1
                epoch_count = 0
                for epoch in np.arange(self.n_epochs):
                    for i, (Iindices, Jindices, Xij) in enumerate(self.generate_batches()):
                        clock.update()
                        feed_dict = {
                            model.indexI: Iindices,
                            model.indexJ: Jindices,
                            model.Xij: Xij
                        }
                        _, loss, lr, gs = self.session.run([model.optimizer,
                                                       model.loss,
                                                       model.learnRate,
                                                       model.global_step],
                                                      feed_dict=feed_dict)
                        loss = loss / self.batch_size
                        progression = gs / self.num_batches
                        if gs % (self.num_batches // 1000) == 0:
                            now = time.strftime("%H:%M:%S")
                            logger.debug("{} Progression={:1} Ep={:2}  GS={:5.2e}  LR={:5.2e}  Loss={:4.3f}  Speed={:5.2e}s/sec".format(
                                now, int(progression*100), epoch, gs, lr, loss,  self.batch_size * gs / (time.time() - start)))
                        if self.saver is not None and progression >= saver_step:
                            saver_step += 0.1
                            emb_types = ["IJ", "I", "J"]
                            for e in emb_types:
                                self.I_matrix = self.graph.get_tensor_by_name("IW:0").eval()
                                self.J_matrix = self.graph.get_tensor_by_name("JW:0").eval()
                                self.I_vectors = self.I_matrix
                                self.J_vectors = self.J_matrix
                                self.saver(self.get_embeddings(e), e + "_" + str(gs % (self.num_batches // 10)))
                        if self.callback is not None and epoch == epoch_count:
                            epoch_count += 1
                            self.I_matrix = self.graph.get_tensor_by_name("IW:0").eval()
                            self.J_matrix = self.graph.get_tensor_by_name("JW:0").eval()
                            self.I_vectors = self.I_matrix
                            self.J_vectors = self.J_matrix
                            embI = self.get_embeddings("I")
                            embJ = self.get_embeddings("J")
                            embIJ = self.get_embeddings("IJ")
                            self.callback(embI, embJ, embIJ, epoch, loss)

        with self.session.as_default():
            self.I_matrix = self.graph.get_tensor_by_name("IW:0").eval()
            self.J_matrix = self.graph.get_tensor_by_name("JW:0").eval()
            self.I_vectors = self.I_matrix
            self.J_vectors = self.J_matrix
        if self.saver is not None:
            emb_types = ["IJ", "I", "J"]
            for e in emb_types:
                self.saver(self.get_embeddings(e), e)
        if self.callback is not None:
            embI = self.get_embeddings("I")
            embJ = self.get_embeddings("J")
            embIJ = self.get_embeddings("IJ")
            self.callback(embI, embJ, embIJ, self.n_epochs, loss)

    def get_embeddings(self, embedding='IJ'):
        if embedding == 'IJ':
            return np.hstack([self.I_vectors, self.J_vectors])
        elif embedding == 'I':
            return self.I_vectors
        elif embedding == 'J':
            return self.J_vectors

    def get_matrix(self, matrix='IJ'):
        if matrix == 'IJ':
            return np.vstack([self.I_matrix, self.J_matrix])
        elif matrix == 'I':
            return self.I_matrix
        elif matrix == 'J':
            return self.J_matrix
