
import numpy as np
import logging
import pkg_resources
import scipy.io

logger = logging.getLogger()


def get_dataset(dataset):
    available = ["cora", "aminer", "citeseer", "flickr", "blogcatalog", "wikipedia", "ppi"]
    if dataset not in available:
        logger.warning("Dataset %s doesn't exist.",dataset)
        logger.info("Please select one of these: %s", available)
    logger.debug("Getting %s", dataset)

    data = np.load(pkg_resources.resource_filename("gvnr", f"resources/{dataset}.npz"))

    for k in data:
        logger.debug(f"Shape of {k}: {data[k].shape}")

    binary_vectors = data["binary_vectors"]
    tfidf_vectors = data["tfidf_vectors"]
    svd_vectors = data["svd_vectors"]
    gt_mask = data["gt_mask"]
    adjacency_matrix = data["adjacency_matrix"]
    labels = data["labels"]

    return binary_vectors.astype(np.int), tfidf_vectors.astype(np.float), svd_vectors.astype(np.float), adjacency_matrix.astype(np.float), labels.astype(np.int), gt_mask.astype(np.int)

