from sklearn.preprocessing import normalize
import gvnr.models.tadw
import gvnr.models.netmf
import gvnr.models.gvnr
import gvnr.models.gvnrt
import gvnr.models.glove

import gvnr.models.deepwalk

import numpy as np
import gvnr.preprocessing.random_walker
import gvnr.preprocessing.window_slider
import sklearn.preprocessing

import scipy.sparse

import logging
logger = logging.getLogger()

number_of_walks = 80
num_neg = 1
x_min = 1
batch_size = 128


class binary_wrapper(object):
    def __init__(self):
        pass

    def fit_predict(self, features, adjacency_matrix):
        return features


class tfidf_wrapper(object):
    def __init__(self):
        pass

    def fit_predict(self, features, adjacency_matrix):
        return features


class svd_wrapper(object):
    def __init__(self):
        pass

    def fit_predict(self, features, adjacency_matrix):
        return normalize(features)



class tadw_wrapper(object):
    def __init__(self):
        self.tadw = gvnr.models.tadw.TADW()

    def fit_predict(self, features, adjacency_matrix):
        self.tadw.fit(features, adjacency_matrix)
        I = normalize(self.tadw.get_embeddings('I'))
        J = normalize(self.tadw.get_embeddings('J'))
        tadw_vectors = normalize(np.hstack((I, J)), axis=0, norm='l2')
        return tadw_vectors


class netmf_wrapper(object):
    def __init__(self):
        pass

    def fit_predict(self, features, adjacency_matrix):
        return gvnr.models.netmf.netmf_large(adjacency_matrix)

class netmf_small_wrapper(object):
    def __init__(self):
        pass

    def fit_predict(self, features, adjacency_matrix):
        return gvnr.models.netmf.netmf_small(adjacency_matrix)

class deepwalk_wrapper(object):
    def __init__(self):
        pass

    def fit_predict(self, features, adjacency_matrix):
        return gvnr.models.deepwalk.run(adjacency_matrix)

class deepwalk_svd_wrapper(object):
    def __init__(self):
        pass

    def fit_predict(self, features, adjacency_matrix):
        deepwalk_vectors = gvnr.models.deepwalk.run(adjacency_matrix)
        deepwalk_vectors = normalize(deepwalk_vectors)
        svd_vectors = normalize(features)
        return normalize(np.hstack((deepwalk_vectors, svd_vectors)), axis=0)


class netmf_svd_wrapper(object):
    def __init__(self):
        pass

    def fit_predict(self, features, adjacency_matrix):
        netmf_vectors = gvnr.models.netmf.netmf_large(adjacency_matrix)
        netmf_vectors = normalize(netmf_vectors)
        svd_vectors = normalize(features)
        return normalize(np.hstack((netmf_vectors, svd_vectors)), axis=0)


class glove_wrapper(object):
    def __init__(self, X=None):
        self.X = X

    def fit_predict(self, features, adjacency_matrix):
        adjacency_matrix = scipy.sparse.csr_matrix(adjacency_matrix)
        nodes_number = adjacency_matrix.shape[0]
        X = None
        if self.X is None:
            random_walker = gvnr.preprocessing.random_walker.RandomWalker(
                adjacency_matrix,
                walks_length=40,
                walks_number=number_of_walks
            )
            random_walks = random_walker.build_random_walks()
            slider = gvnr.preprocessing.window_slider.WindowSlider(
                random_walks,
                nodes_number,
                window_size=5,
                window_factor="decreasing"
            )
            X = slider.build_cooccurrence_matrix()
        else:
            X = self.X

        model = gvnr.models.glove.GloVe()

        model.fit(X,
                  learn_rate=0.001,
                  embedding_size=80,
                  batch_size=batch_size,
                  n_epochs=10
                  )
        I = normalize(model.get_embeddings(embedding='I')+model.get_embeddings(embedding='J'), axis=0)
        return I


class gvnr_no_filter_wrapper(object):
    def __init__(self, X=None):
        self.X = X

    def fit_predict(self, features, adjacency_matrix):
        adjacency_matrix = scipy.sparse.csr_matrix(adjacency_matrix)
        nodes_number = adjacency_matrix.shape[0]
        X = None
        if self.X is None:
            random_walker = gvnr.preprocessing.random_walker.RandomWalker(
                adjacency_matrix,
                walks_length=40,
                walks_number=number_of_walks
            )
            random_walks = random_walker.build_random_walks()
            slider = gvnr.preprocessing.window_slider.WindowSlider(
                random_walks,
                nodes_number,
                window_size=5,
                window_factor="decreasing"
            )
            X = slider.build_cooccurrence_matrix()
        else:
            X = self.X

        model = gvnr.models.gvnr.gvnr()

        model.fit(X,
                  learn_rate=0.001,
                  embedding_size=80,
                  batch_size=batch_size,
                  n_epochs=10,
                  k_neg=num_neg,
                  x_min=0
                  )

        I = normalize(model.get_embeddings(embedding='I') + model.get_embeddings(embedding='J'), axis=0)
        return I


class gvnr_wrapper(object):
    def __init__(self, X=None):
        self.X = X

    def fit_predict(self, features, adjacency_matrix):
        adjacency_matrix = scipy.sparse.csr_matrix(adjacency_matrix)
        nodes_number = adjacency_matrix.shape[0]
        X = None
        if self.X is None:
            random_walker = gvnr.preprocessing.random_walker.RandomWalker(
                adjacency_matrix,
                walks_length=40,
                walks_number=number_of_walks
            )
            random_walks = random_walker.build_random_walks()
            slider = gvnr.preprocessing.window_slider.WindowSlider(
                random_walks,
                nodes_number,
                window_size=5,
                window_factor="decreasing"
            )
            X = slider.build_cooccurrence_matrix()
        else:
            X = self.X

        model = gvnr.models.gvnr.gvnr()

        model.fit(X,
                              learn_rate=0.001,
                              embedding_size=80,
                              batch_size=batch_size,
                              n_epochs=2,
                              k_neg=num_neg,
                              x_min=x_min
                              )

        I = normalize(model.get_embeddings(embedding='I') + model.get_embeddings(embedding='J'), axis=0)
        return I

class gvnrt_wrapper(object):
    def __init__(self, X=None):
        self.X = X

    def fit_predict(self, features, adjacency_matrix, pretrained_word_embeddings=None):
        adjacency_matrix = scipy.sparse.csr_matrix(adjacency_matrix)
        features = scipy.sparse.csr_matrix(features)
        nodes_number = adjacency_matrix.shape[0]
        X = None
        if self.X is None:
            random_walker = gvnr.preprocessing.random_walker.RandomWalker(
                adjacency_matrix,
                walks_length=40,
                walks_number=number_of_walks
            )
            random_walks = random_walker.build_random_walks()
            slider = gvnr.preprocessing.window_slider.WindowSlider(
                random_walks,
                 nodes_number,
                 window_size=5,
                 window_factor="decreasing"
            )
            X = slider.build_cooccurrence_matrix()
        else:
            X = self.X

        model = gvnr.models.gvnrt.gvnrt()

        model.fit(X,
                  features,
                  pretrained_word_embeddings=pretrained_word_embeddings,
                  learn_rate=0.001,
                  embedding_size=80,
                  batch_size=batch_size,
                  n_epochs=4,
                  k_neg=num_neg,
                  x_min=x_min
                  )
        I = normalize(model.get_embeddings(embedding='I'), norm='l2')
        J = normalize(model.get_embeddings(embedding='J'), norm='l2')
        IJ = normalize(np.hstack([I, J]), axis=0, norm='l2')
        return IJ
