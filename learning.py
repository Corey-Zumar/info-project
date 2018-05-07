import sys
import os
import argparse
import numpy as np
import math
import time
import pickle

from multiprocessing import Process

from graph_utils import construct_network_graph
from graph_utils import method_common_neighbors, method_jaccard_coeff

from parallelism import get_worker_range_pairs

NUM_DATASET_WORKERS = 64

INPUT_VECTOR_SIZE = 1154
NUM_CLASSES = 2

TRAINING_LABEL_POSITIVE = 1
TRAINING_LABEL_NEGATIVE = 0

def create_feature_vector(node_1, node_2):
    node_1_features = node_1.get_features()
    node_2_features = node_2.get_features()
    assert node_1_features is not None and node_2_features is not None

    feature_vec = np.append(node_1_features, node_2_features)
    common_neighbors = method_common_neighbors(node_1, node_2)
    jaccard_coeff = method_jaccard_coeff(node_1, node_2)
    feature_vec = np.append(feature_vec, [common_neighbors, jaccard_coeff])

    return feature_vec

def construct_training_dataset(graph, output_dir, num_workers):
    node_ids = list(graph.get_node_ids())

    def get_label(node_1, node_2):
        if node_1.get_id() in node_2.get_connection_ids():
            return TRAINING_LABEL_POSITIVE
        else:
            return TRAINING_LABEL_NEGATIVE

    def eval_helper(graph, node_ids, r1_low, r1_high, r2_low, r2_high, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        examined = set()

        training_data = {
                            TRAINING_LABEL_POSITIVE : [],
                            TRAINING_LABEL_NEGATIVE : []
                        }

        for i in xrange(r1_low, r1_high):
            i_node_id = node_ids[i]
            i_node = graph.get_node(i_node_id)
            for j in xrange(r2_low, r2_high):
                j_node_id = node_ids[j]
                results_key = frozenset((i_node_id, j_node_id))

                if results_key in examined or i_node_id == j_node_id:
                    continue

                examined.add(results_key)

                j_node = graph.get_node(j_node_id)

                i_node_features = i_node.get_features()
                j_node_features = j_node.get_features()

                if i_node_features is None or j_node_features is None:
                    continue

                # Concatenate the features of each node to get a feature
                # vector that will be used for developing a classifier
                feature_vec = create_feature_vector(i_node, j_node)

                label = get_label(i_node, j_node)
                training_data[label].append(feature_vec)

        data_subpath = "{r1l}_{r1h}_{r2l}_{r2h}_data.pkl".format(r1l=r1_low,
                                                                 r1h=r1_high,
                                                                 r2l=r2_low,
                                                                 r2h=r2_high)
        pos_data_subpath = "pos_" + data_subpath
        neg_data_subpath = "neg_" + data_subpath

        pos_data_path = os.path.join(output_dir, pos_data_subpath)
        with open(pos_data_path, "w") as f:
            pickle.dump(training_data[TRAINING_LABEL_POSITIVE], f)

        print("Wrote positive training data to file with path: {}".format(pos_data_path))

        neg_data_path = os.path.join(output_dir, neg_data_subpath)
        with open(neg_data_path, "w") as f:
            pickle.dump(training_data[TRAINING_LABEL_NEGATIVE], f)

        print("Wrote negative training data to file with path: {}".format(neg_data_path))

    range_pairs = get_worker_range_pairs(node_ids, num_workers)
    result_procs = []

    total = 0
    range_idx = 0 
    while range_idx < len(range_pairs):
        r1, r2 = range_pairs[range_idx]
        r1_low, r1_high = r1
        r2_low, r2_high = r2

        result_proc = Process(target=eval_helper, args=(graph, node_ids, r1_low, r1_high, r2_low, r2_high, output_dir))
        result_proc.start()
        result_procs.append(result_proc)

        if len(result_procs) >= num_workers:
            for proc in result_procs:
                proc.join()
            result_procs = []

        range_idx += 1

    for proc in result_procs:
        proc.join()

def load_training_data(data_dir_path, max_pos_size, max_neg_size):
    training_data = { TRAINING_LABEL_POSITIVE : [], TRAINING_LABEL_NEGATIVE : []}
    data_fpaths = [os.path.join(data_dir_path, fpath) for fpath in os.listdir(data_dir_path) if fpath.endswith(".pkl")]
    for fpath in data_fpaths:
        try:
            if "pos" in fpath and len(training_data[TRAINING_LABEL_POSITIVE]) >= max_pos_size:
                continue
            if "neg" in fpath and len(training_data[TRAINING_LABEL_NEGATIVE]) >= max_neg_size:
                continue

            with open(fpath, "r") as f:
                feature_vecs = pickle.load(f)

                if "pos" in fpath: 
                    training_data[TRAINING_LABEL_POSITIVE] += feature_vecs 
                elif "neg" in fpath:
                    training_data[TRAINING_LABEL_NEGATIVE] += feature_vecs 
                else:
                    raise

        except Exception as e:
            print(fpath)
            os._exit(0)

    np.random.shuffle(training_data[TRAINING_LABEL_POSITIVE])
    np.random.shuffle(training_data[TRAINING_LABEL_NEGATIVE])

    training_data[TRAINING_LABEL_POSITIVE] = training_data[TRAINING_LABEL_POSITIVE][:max_pos_size]
    training_data[TRAINING_LABEL_NEGATIVE] = training_data[TRAINING_LABEL_NEGATIVE][:max_neg_size]

    return training_data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a training dataset that can be used to train a link predictor')
    parser.add_argument('-p', '--path', type=str, help='Path to the facebook combined graph file')
    parser.add_argument('-f', '--features_path', type=str, help='Path to a directory containing feature vectors for each node')
    parser.add_argument('-o', '--output_path', type=str, help="The output directory path to which to save the pickled training data")
    parser.add_argument('-e', '--ego_node_id', type=int, help='The id of the ego node from which to derive features for the graph')
    parser.add_argument('-l', '--load', action='store_true', help="If specified, load the training data specified by the 'output_path' flag")
    parser.add_argument('-pm', '--pos_max', type=int, help="The maximum number of positive elements in the loaded training dataset")
    parser.add_argument('-nm', '--neg_max', type=int, help="The maximum number of negative elements in the loaded training dataset")

    args = parser.parse_args()

    if args.load:
        load_training_data(args.output_path, args.pos_max, args.neg_max)
    else:
        graph = construct_network_graph(args.path, args.features_path, args.ego_node_id)
        construct_training_dataset(graph, args.output_path, NUM_DATASET_WORKERS)

