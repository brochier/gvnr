from context import gvnr

import os
import sys

import gvnr.baselines

import logging
logger = logging.getLogger()

logger.info("Testing gvnr.baselines...")

list_graph_text = [
    "binary",
    "tfidf",
    "svd",
    "netmf",
    "netmf_small",
    "deepwalk",
    "deepwalk_svd",
    "netmf_svd",
    "tadw",
    "glove",
    "gvnr_no_filter",
    "gvnr",
    "gvnr"
]

list_graph_only = [
    "netmf",
    "netmf_small",
    "deepwalk",
    "glove",
    "gvnr_no_filter",
    "gvnr"
]

list_graph_text = [
    "svd"
]

list_graph_only = [
    "netmf_small"
]

baselines = {
    "cora": list_graph_text,
    "citeseer": list_graph_text,
    "aminer": list_graph_only,
    "wikipedia": list_graph_only,
    "ppi": list_graph_only,
    "blogcatalog": list_graph_only,
    "flickr": list_graph_only
}

gvnr.baselines.run(baselines)

logger.info("Test ok !")
