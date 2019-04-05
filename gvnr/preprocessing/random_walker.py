import numpy as np
import os
import sklearn.preprocessing
import multiprocessing
from functools import partial
import tempfile
import time

import logging
logger = logging.getLogger()

"""
RandomWalker: class to generate sequences of nodes based on random walks from an adjency matrix
"""
class RandomWalker:
    def __init__(self, adjacency_matrix, walks_length, walks_number):
        self.adjacency_matrix = adjacency_matrix # scipy.sparse.csr_matrix
        self.walks_length = walks_length # int
        self.walks_number = walks_number # int

    """
    build_random_walks: 
    """
    def build_random_walks(self, gather_walks = True):
        with tempfile.TemporaryDirectory() as dump_dir:
            jobs = []
            seed = int(time.time())
            for walk_index in range(self.walks_number):
                np.random.seed(seed+walk_index)
                random_choice = RandomChoice(self.adjacency_matrix)
                func = partial(one_walk,
                               walks_length = self.walks_length,
                               random_choice = random_choice,
                               dump_dir = dump_dir)
                p = multiprocessing.Process(target=func,
                                            name=multiprocessing.current_process().name,
                                            args=(walk_index,))
                p.name = multiprocessing.current_process().name
                jobs.append(p)
                p.start()
            randoms_walks = list()
            for walk_index in range(self.walks_number):
                jobs[walk_index].join()
                if gather_walks is True:
                    logger.debug(" ".join([str(v) for v in ["Adding walk index", walk_index, "to final random walks list."]]))
                    current_random_walk = np.load(os.path.join(dump_dir, "random_walk_{0}.npy".format(walk_index)))
                    randoms_walks.append(current_random_walk)
            randoms_walks = np.vstack(randoms_walks)
            logger.debug(" ".join([str(v) for v in ["Final number of walks: ", randoms_walks.shape[0]]]))
            return randoms_walks

    def random_walks_generator(self):
        random_choice = RandomChoice(self.adjacency_matrix)
        N = random_choice.nodes_number
        for i in range(self.walks_number):
            starting_nodes = np.arange(N)
            np.random.shuffle(starting_nodes)
            for start_node in starting_nodes:
                walker_positions = np.zeros(self.walks_length + 1, dtype=np.int32)
                walker_positions[0] = start_node
                for j in range(self.walks_length):
                    next_node = random_choice[walker_positions[j]]
                    walker_positions[j+1] = next_node
                yield walker_positions

        
"""
RandomChoice: class to generate per node walk choices
"""
class RandomChoice(object):
    def __init__(self, adjacency_matrix):
        # Make sure nodes don't point to themselves
        adjacency_matrix.setdiag(0)
        adjacency_matrix.eliminate_zeros()

        self.nodes_number = adjacency_matrix.shape[0]

        # Normalize each row to get transition probabilities for each node
        transition_matrix = sklearn.preprocessing.normalize(adjacency_matrix, axis=1, norm='l1', copy=False)
        # tuple of indices ([row, row, row], [col, col, col])
        nonzero = transition_matrix.nonzero()
        # Corresponding values of previous (row,col) pairs
        data = transition_matrix.data

        K = len(data)
        choices = list()
        probs = list()
        k = 0
        for i in range(self.nodes_number):
            choices.append(list())
            probs.append(list())
            if k >= K: # if we finished looping on data, we fill the remaining nodes transitions
                choices[i] = [-1]
                probs[i] = [1]
                continue
            elif nonzero[0][k] > i: # if the current node has no transition
                choices[i] = [-1]
                probs[i] = [1]
            else:
                while k < K and nonzero[0][k] == i: # loop over currrent node transitions indices/probabilities
                    choices[i].append(nonzero[1][k])
                    probs[i].append(data[k])
                    k += 1
            choices[i] = np.array(choices[i], dtype=np.int32)
            probs[i] = np.array(probs[i], dtype=np.float32)
        choices.append( np.array([-1], dtype=np.int32)) # Add -1 choice
        probs.append( np.array([1], dtype=np.int32))
        self.choices = choices
        self.prob = probs

    def __getitem__(self, arg):
        # return a randomly selected neighbor from node "arg"
        return np.random.choice(self.choices[arg], p=self.prob[arg])

"""
Create sequences of node starting RW from each node
"""
def one_walk(walk_index, walks_length, random_choice, dump_dir):
    logger.debug(" ".join([str(v) for v in ["Starting walk index", walk_index]]))
    N = random_choice.nodes_number
    walkers_positions = np.zeros((N,walks_length + 1), dtype=np.int32)
    starting_nodes = np.arange(N)
    np.random.shuffle(starting_nodes)
    for start_node in starting_nodes:
        walkers_positions[start_node,0] = start_node
        # Random walks
        for j in range(walks_length):
            next_node = random_choice[walkers_positions[start_node,j]]
            walkers_positions[start_node,j+1] = next_node
    np.save(os.path.join(dump_dir, "random_walk_{0}".format(walk_index)), walkers_positions)
    return walk_index
