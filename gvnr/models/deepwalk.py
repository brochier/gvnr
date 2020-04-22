import logging

logger = logging.getLogger()
logging.getLogger("gensim").setLevel(logging.WARNING)

from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from itertools import permutations
from scipy.sparse import issparse
from io import open
from time import time
import random
from six.moves import zip
from gensim.models import Word2Vec
import scipy.sparse
from sklearn.preprocessing import normalize
import numpy as np





class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):

        t0 = time()

        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        logger.debug('make_directed: added missing edges {}s'.format(t1 - t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        logger.debug('make_consistent: made consistent in {}s'.format(t1 - t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()

        logger.debug('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return self.order()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1, size + 1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def parse_adjacencylist(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            introw = [int(x) for x in l.strip().split()]
            row = [introw[0]]
            row.extend(set(sorted(introw[1:])))
            adjlist.extend([row])

    return adjlist


def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist


def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)

    G.make_consistent()
    return G

def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors
    return G

class WalksCorpus(object):
  def __init__(self, file_list):
    self.file_list = file_list
  def __iter__(self):
    for file in self.file_list:
      with open(file, 'r') as f:
        for line in f:
          yield line.split()


def run(adjacency_matrix, embedding_size=80):
    logger.debug("Walking Deepwalk...")
    adjacency_matrix = scipy.sparse.csr_matrix(adjacency_matrix)
    G = from_numpy(adjacency_matrix)
    walks = build_deepwalk_corpus(G, num_paths=80, path_length=40, alpha=0, rand=random.Random())
    logger.debug("Training Deepwalk...")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)
    model = Word2Vec(walks, size=160, window=5, min_count=0, sg=1, hs=0, negative=10, workers=10)
    vectors = np.zeros((adjacency_matrix.shape[0], 160))
    for i in range(adjacency_matrix.shape[0]):
        vectors[i] = model.wv.get_vector(str(i))
    return normalize(vectors, axis=0)