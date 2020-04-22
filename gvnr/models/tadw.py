import numpy as np
from sklearn.preprocessing import normalize

import logging

logger = logging.getLogger()


class TADW(object):

    def __init__(self, dim=80, lamb=0.2):
        self.lamb = lamb
        self.dim = dim
        self.embeddings = None
        self.I = None
        self.J = None

    def get_embeddings(self, type):
        if type == 'I':
            return self.I
        elif type == 'J':
            return self.J

    def fit(self, svd_vectors, adjacency_matrix):
        self.adj = adjacency_matrix
        self.adj = self.adj / self.adj.sum(axis=1)
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        self.M = (self.adj + np.dot(self.adj, self.adj)) / 2
        # T is feature_size*node_num, text features
        self.T = svd_vectors.T
        self.node_size = self.adj.shape[0]
        self.feature_size = self.T.shape[0]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)
        # Update
        for i in range(20):
            logger.debug('Iteration %s', i)
            # Update W
            B = np.dot(self.H, self.T)
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                  2 * np.dot(B, self.M.T) + self.lamb * self.W
            Hess = 2 * np.dot(B, B.T) + self.lamb * np.eye(self.dim)
            drv = np.reshape(drv, [self.dim * self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim * self.node_size, 1])
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [self.dim * self.node_size, 1])

                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecW = vecW + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
                          - np.dot(self.W, self.M.T)), self.T.T) + self.lamb * self.H
            drv = np.reshape(drv, (self.dim * self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim * self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                 + self.lamb * dtS, (self.dim * self.feature_size, 1))
                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecH = vecH + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))

        self.I = self.W.T # graph
        self.J = np.dot(self.T.T, self.H.T) # text

    def predict(self, T):
        J = normalize(np.dot(T, self.H.T))
        return J


"""
self.adj = adjacency_matrix
        self.adj = normalize(self.adj, norm='l1')
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        self.M = (self.adj + np.dot(self.adj, self.adj)) / 2
        # T is feature_size*node_num, text features
        self.T = svd_vectors.T
        self.node_size = self.adj.shape[0]
        self.feature_size = self.T.shape[0]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)

        # Update
        for i in range(num_iterations):
            logger.debug('Iteration %s', i)
            # Update W
            B = np.dot(self.H, self.T)
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                  2 * np.dot(B, self.M.T) + self.lamb * self.W
            Hess = 2 * np.dot(B, B.T) + self.lamb * np.eye(self.dim)
            drv = np.reshape(drv, [self.dim * self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim * self.node_size, 1])
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [
                    self.dim * self.node_size, 1])

                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecW = vecW + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
                          - np.dot(self.W, self.M.T)), self.T.T) + self.lamb * self.H
            drv = np.reshape(drv, (self.dim * self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim * self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                 + self.lamb * dtS, (self.dim * self.feature_size, 1))
                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecH = vecH + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))
"""
