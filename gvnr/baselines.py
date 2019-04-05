import os

import pkg_resources
import scipy.sparse

import gvnr.evaluation
import gvnr.data.datasets
import gvnr.models.wrappers

import gvnr.preprocessing.random_walker
import gvnr.preprocessing.window_slider

import numpy as np

import logging
logger = logging.getLogger()

proportions = [0.1, 0.2, 0.3, 0.4, 0.5]

def run(baselines, save = False):
    logger.info("Running evaluations with parameters:")
    for d in baselines:
        logger.info(f"{d}: {baselines[d]}")

    for dataset in baselines:
        binary_vectors, tfidf_vectors, svd_vectors, adjacency_matrix, labels, gt_mask = gvnr.data.datasets.get_dataset(
            dataset)
        X_file = pkg_resources.resource_filename("gvnr", 'resources/{0}_X.npz'.format(dataset))
        X = None
        if os.path.isfile(X_file):
            X = scipy.sparse.load_npz(X_file)
        else:
            random_walker = gvnr.preprocessing.random_walker.RandomWalker(
                scipy.sparse.csr_matrix(adjacency_matrix),
                walks_length=40,
                walks_number=80
            )
            random_walks = random_walker.build_random_walks()
            slider = gvnr.preprocessing.window_slider.WindowSlider(
                random_walks,
                adjacency_matrix.shape[0],
                window_size=5,
                window_factor="decreasing"
            )
            X = slider.build_cooccurrence_matrix()
            scipy.sparse.save_npz(X_file, X)

        for baseline in baselines[dataset]:
            model = None
            features = None
            if baseline == "binary":
                model = gvnr.models.wrappers.binary_wrapper()
                features = binary_vectors
            if baseline == "tfidf":
                model = gvnr.models.wrappers.tfidf_wrapper()
                features = tfidf_vectors
            if baseline == "svd":
                model = gvnr.models.wrappers.svd_wrapper()
                features = svd_vectors
            if baseline == "tadw":
                model = gvnr.models.wrappers.tadw_wrapper()
                features = svd_vectors
            if baseline == "netmf":
                model = gvnr.models.wrappers.netmf_wrapper()
            if baseline == "netmf_small":
                model = gvnr.models.wrappers.netmf_small_wrapper()
            if baseline == "deepwalk":
                model = gvnr.models.wrappers.deepwalk_wrapper()
                features = None
            if baseline == "deepwalk_svd":
                model = gvnr.models.wrappers.deepwalk_svd_wrapper()
                features = svd_vectors
            if baseline == "netmf_svd":
                model = gvnr.models.wrappers.netmf_svd_wrapper()
                features = svd_vectors
            if baseline == "glove":
                model = gvnr.models.wrappers.glove_wrapper(X = X)
                features = binary_vectors
            if baseline == "gvnr_no_filter":
                model = gvnr.models.wrappers.gvnr_no_filter_wrapper(X = X)
                features = binary_vectors
            if baseline == "gvnr":
                model = gvnr.models.wrappers.gvnr_wrapper(X = X)
                features = binary_vectors
            if baseline == "gvnrt":
                model = gvnr.models.wrappers.gvnrt_wrapper(X = X)
                features = binary_vectors

            vectors = model.fit_predict(features, adjacency_matrix)
            if save is True:
                filename = os.path.abspath(
                    os.path.join(os.path.dirname(__file__),
                    os.path.pardir,
                    f"embeddings/{dataset}_{baseline}_vectors.csv")
                    )
                logger.info(f"Saving embeddings to {filename}...")
                np.savetxt(filename, vectors)

            vector = vectors[gt_mask]
            scores = None
            if dataset in ["cora", "aminer", "wiki", "citeseer"]:
                scores = gvnr.evaluation.get_score(vectors, labels)
            if dataset in ["flickr", "blogcatalog", "wikipedia", "ppi"]:
                scores = gvnr.evaluation.get_score(vectors, labels, multilabels=True)
            logger.info("DATASET: {0}, BASELINE: {1} => f1_micro {2} f1_macro {2}".format(dataset, baseline, proportions))
            logger.info("   ".join(["&{0:.1f}".format(s*100) for s in list(scores["f1_micro"]) + list(scores["f1_macro"]) ] ))
