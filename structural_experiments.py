import sys
import os
import argparse
import itertools
import math
import json
import numpy as np

from datetime import datetime

import graph_utils

from graph_utils import RESULTS_KEY_SENSITIVITY, RESULTS_KEY_SPECIFICITY 
from graph_utils import RESULTS_KEY_PRECISION, RESULTS_KEY_TOTAL_EXPLORED
from graph_utils import RESULTS_KEY_FALSE_POSITIVE_RATE

EXPERIMENT_METHOD_COMMON_NEIGHBORS = "common_neighbors"
EXPERIMENT_METHOD_JACCARD_COEFF = "jaccard"
EXPERIMENT_METHOD_PREF_ATTACHMENT = "pref_attachment"
EXPERIMENT_METHOD_ALWAYS_ZERO = "always_zero"

RESULTS_KEY_METHOD_NAME = "method_name"
RESULTS_KEY_THRESHOLD = "threshold"
RESULTS_KEY_GRAPH_PATH = "graph_path"

NUM_EXPERIMENT_WORKERS = 64

def save_results(graph_path, method_name, threshold, results_dir, stats_results):
    sensitivity = stats_results[RESULTS_KEY_SENSITIVITY]
    specificity = stats_results[RESULTS_KEY_SPECIFICITY]
    precision = stats_results[RESULTS_KEY_PRECISION]
    false_positive_rate = stats_results[RESULTS_KEY_FALSE_POSITIVE_RATE]
    total_explored = stats_results[RESULTS_KEY_TOTAL_EXPLORED]

    results_json = {
        RESULTS_KEY_SENSITIVITY : sensitivity,
        RESULTS_KEY_SPECIFICITY : specificity,
        RESULTS_KEY_PRECISION : precision,
        RESULTS_KEY_FALSE_POSITIVE_RATE : false_positive_rate,
        RESULTS_KEY_TOTAL_EXPLORED : total_explored,
        RESULTS_KEY_METHOD_NAME : method_name,
        RESULTS_KEY_THRESHOLD : threshold,
        RESULTS_KEY_GRAPH_PATH : graph_path
    }

    results_fname = "results_{mn}_{thresh}_{:%y%m%d_%H%M%S}.json".format(datetime.now(),
                                                                         mn=method_name,
                                                                         thresh=threshold)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_file = os.path.join(results_dir, results_fname)
    with open(results_file, "w") as f:
        json.dump(results_json, f, indent=4)

def get_method_fn(method_name):
    if method_name == EXPERIMENT_METHOD_COMMON_NEIGHBORS:
        return graph_utils.method_common_neighbors
    elif method_name == EXPERIMENT_METHOD_JACCARD_COEFF:
        return graph_utils.method_jaccard_coeff
    elif method_name == EXPERIMENT_METHOD_PREF_ATTACHMENT:
        return graph_utils.method_pref_attachment
    elif method_name == EXPERIMENT_METHOD_ALWAYS_ZERO:
        return graph_utils.method_always_zero
    else:
        raise Exception("Invalid method name: {}".format(method_name))

def main(graph_path, method_name, threshold, results_dir):
    graph = graph_utils.construct_network_graph(graph_path)
    method_fn = get_method_fn(method_name)
    scorer = graph_utils.create_scorer(method_fn, threshold)
    stats_results = graph_utils.eval_link_prediction_method(graph, scorer, NUM_EXPERIMENT_WORKERS)

    save_results(graph_path, 
                 method_name, 
                 threshold, 
                 results_dir, 
                 stats_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments on facebook graphs')
    parser.add_argument('-g', '--graph_path', type=str, help='Path to the facebook combined graph file')
    parser.add_argument('-m', '--method_name', type=str, help='The name of the evaluation method to use when scoring the graph')
    parser.add_argument('-t', '--thresholds', type=float, nargs='+', help='The score thresholds required to designate a positive result')
    parser.add_argument('-r', '--results_dir', type=str, help='The path of the directory to which to write results')

    args = parser.parse_args()

    for threshold in args.thresholds:
        main(args.graph_path, 
             args.method_name,
             threshold,
             args.results_dir)


