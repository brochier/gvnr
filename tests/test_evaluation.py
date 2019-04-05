from context import gvnr
import gvnr.data.datasets
import gvnr.evaluation

import os
import pkg_resources

from sklearn.preprocessing import normalize



import logging
logger = logging.getLogger()

current_folder = os.path.dirname(os.path.abspath(__file__))
logger.info("Testing gvnr.evaluate...")

proportions = [0.1, 0.2, 0.3, 0.4, 0.5]

datasets = ["cora"]


for dataset in datasets:
    binary_vectors, tfidf_vectors, svd_vectors, adjacency_matrix, labels, gt_mask = gvnr.data.datasets.get_dataset(dataset)
    scores = gvnr.evaluation.get_score(normalize(svd_vectors), labels)
    for k in scores:
        print_scores = " | ".join(["{0} => {1:.3f}".format(a, b) for a, b in zip(proportions, scores[k])])
        logger.info("METRIC: %s, SCORES: %s", k, print_scores)

logger.info("Test ok !")
