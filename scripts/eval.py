from context import gvnr
import sys
import gvnr.baselines
import argparse
import logging
logger = logging.getLogger()

list_methods_graph_text = [
    "binary",
    "tfidf",
    "svd",
    "deepwalk_svd",
    "netmf_svd",
    "tadw",
    "gvnrt"
]

list_methods_graph_only = [
    "netmf",
    "netmf_small",
    "deepwalk",
    "glove",
    "gvnr_no_filter",
    "gvnr"
]

list_dataset_graph_text = [
    "cora",
    "citeseer"
]

list_dataset_graph_only = [
    "aminer",
    "wikipedia",
    "ppi",
    "blogcatalog"
]

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--save",
                    help="If specified, saves the computed embeddings (csv format) in the 'gnvr/embeddings/' directory.",
                    action="store_true")
parser.add_argument("-m", "--method", type=str, help=f"Method to be used. Each of them if not specified."
                                                     f" Available options are (graph+text) {list_methods_graph_text}"
                                                     f" and (graph only) {list_methods_graph_only}")
parser.add_argument("-d", "--dataset", type=str, help=f"Dataset to work with. Each of them if not specified."
                                                      f" Available options are (graph+text) {list_dataset_graph_text}"
                                                      f" and (graph only) {list_dataset_graph_only}.")

args = parser.parse_args()


if args.method is not None:
    if args.method in list_methods_graph_text:
        if args.dataset is not None:
            if args.dataset in list_dataset_graph_text:
                baselines = {args.dataset: [args.method]}
                gvnr.baselines.run(baselines, save=args.save)
            elif args.dataset in list_dataset_graph_only:
                logger.error(f"Dataset {args.dataset} (graph only) is not compatible with method {args.method} (graph+text) !")
                sys.exit(0)
            else:
                logger.error(f"Unknown dataset {args.dataset} !")
                sys.exit(0)
        else:
            baselines = {d: [args.method] for d in list_dataset_graph_text}
            gvnr.baselines.run(baselines, save=args.save)
    elif args.method in list_methods_graph_only:
        if args.dataset is not None:
            if args.dataset in list_dataset_graph_only+list_dataset_graph_text:
                baselines = {args.dataset: [args.method]}
                gvnr.baselines.run(baselines, save=args.save)
            else:
                logger.error(f"Unknown dataset {args.dataset} !")
                sys.exit(0)
        else:
            baselines = {d: [args.method] for d in list_dataset_graph_only+list_dataset_graph_text}
            gvnr.baselines.run(baselines, save=args.save)
    else:
        logger.error(f"Unknown method {args.method} !")
        sys.exit(0)
else:
    baselines = {
        "cora": list_methods_graph_only+list_methods_graph_text,
        "citeseer": list_methods_graph_only+list_methods_graph_text,
        "aminer": list_methods_graph_only,
        "wikipedia": list_methods_graph_only,
        "ppi": list_methods_graph_only,
        "blogcatalog": list_methods_graph_only
    }
    if args.dataset is not None:
        if args.dataset in list_dataset_graph_only+list_dataset_graph_text:
            baselines = {args.dataset: baselines[args.dataset]}
            gvnr.baselines.run(baselines, save=args.save)
        else:
            logger.error(f"Unknown dataset {args.dataset} !")
            sys.exit(0)
    else:
        gvnr.baselines.run(baselines, save=args.save)