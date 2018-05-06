import sys
import os
import argparse
import numpy as np

from datetime import datetime
from threading import RLock
from multiprocessing import Process
from multiprocessing import Queue as MPQueue

from featurizer import create_feature_vectors
from parallelism import get_worker_range_pairs

RESULTS_KEY_SENSITIVITY = "sensitivity"
RESULTS_KEY_SPECIFICITY = "specificity"
RESULTS_KEY_PRECISION = "precision"
RESULTS_KEY_FALSE_POSITIVE_RATE = "false_positive_rate"
RESULTS_KEY_TOTAL_EXPLORED = "total_explored"

class Graph:

    def __init__(self):
        self.nodes = {}
        self.nodes_lock = RLock()

    def add_node(self, node):
        self.nodes_lock.acquire()
        if node not in self.nodes:
            self.nodes[node.node_id] = node
            self.nodes_lock.release()
        else:
            self.nodes_lock.release()
            raise Exception("Attempted to add redundant node with id: {}".format(node_id))

    def get_node(self, node_id):
        self.nodes_lock.acquire()
        if node_id in self.nodes:
            node = self.nodes[node_id]
        else:
            node = None

        self.nodes_lock.release()

        return node

    def get_nodes(self, node_ids):
        self.nodes_lock.acquire()
        nodes_mapping = {}
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
            else:
                node = None

            nodes_mapping[node_id] = node

        self.nodes_lock.release()

        return nodes_mapping

    def get_node_ids(self):
        self.nodes_lock.acquire()
        keys = self.nodes.keys()
        self.nodes_lock.release()
        return keys

    def __len__(self):
        return len(self.nodes)

class Node:

    def  __init__(self, node_id, features=None):
        self.node_id = node_id
        # A list of nodes to which the current
        # node is connected via an undirected edge
        self.connections = set()
        self.connection_ids = set()
        self.node_id = node_id
        self.features = features

    def add_connection(self, other_node):
        self.connections.add(other_node)
        self.connection_ids.add(other_node.get_id())

    def connected_to_id(self, node_id):
        result = (node_id in self.connection_ids)
        return result

    def connected_to_node(self, node):
        return self.connected_to_node(node.get_id())

    def get_connection_ids(self):
        return self.connection_ids

    def get_connection(self):
        return self.connections

    def get_id(self):
        return self.node_id

    def get_features(self):
        return self.features

def construct_network_graph(graph_file_path, features_dir_path=None, ego_node_id=None):
    if features_dir_path:
        node_features = create_feature_vectors(features_dir_path, ego_node_id)        
    else:
        node_features = None

    with open(graph_file_path, "r") as f:
        connection_lines = [line.rstrip().split(" ") for line in f.readlines()]

    connections = [(int(source), int(dest)) for source, dest in connection_lines]
    sources, dests = zip(*connections)
    unique_nodes = set(sources + dests)

    network_graph = Graph()

    for node_id in unique_nodes:
        if node_features and node_id in node_features:
            features = node_features[node_id]
        else:
            features = None
        new_node = Node(node_id, features)
        network_graph.add_node(new_node)

    for source_id, dest_id in connections:
        source_dest_nodes = network_graph.get_nodes([source_id, dest_id])
        source_node = source_dest_nodes[source_id]
        dest_node = source_dest_nodes[dest_id]
        source_node.add_connection(dest_node)
        dest_node.add_connection(source_node)

    return network_graph

def method_common_neighbors(node_1, node_2):
    return len(node_1.get_connection_ids().intersection(node_2.get_connection_ids()))

def method_common_neighbors_exclude_selves(node_1, node_2):
    node_1_conn_ids = set(node_1.get_connection_ids())
    node_2_conn_ids = set(node_2.get_connection_ids())
    if node_1.get_id() in node_2_conn_ids:
        # The nodes are connected, so we should
        # remove the entry in a given node's
        # connections list that corresponds to the other node
        node_1_conn_ids.discard(node_2.get_id())
        node_2_conn_ids.discard(node_1.get_id())

    return len(node_1_conn_ids.intersection(node_2_conn_ids))

def method_jaccard_coeff(node_1, node_2):
    len_intersection = len(node_1.get_connection_ids().intersection(node_2.get_connection_ids()))
    len_union = len(node_1.get_connection_ids().union(node_2.get_connection_ids()))

    return float(len_intersection) / len_union

def method_pref_attachment(node_1, node_2):
    return len(node_1.get_connection_ids()) * len(node_2.get_connection_ids())

def method_always_zero(node_1, node_2):
    return 0

def create_scorer(method_fn, correctness_threshold):
    return lambda node_1, node_2 : method_fn(node_1, node_2) >= correctness_threshold

def eval_link_prediction_method(graph, scorer_fn):
    node_ids = list(graph.get_node_ids())

    results_queue = MPQueue()

    def eval_helper(graph, node_ids, r1_low, r1_high, r2_low, r2_high, results_queue):
        try:
            print(r1_low, r1_high, r2_low, r2_high)
            results = set()
            false_negative = 0
            true_negative = 0
            false_positive = 0
            true_positive = 0
            for i in xrange(r1_low, r1_high):
                i_node_id = node_ids[i]
                i_node = graph.get_node(i_node_id)

                for j in xrange(r2_low, r2_high):
                    j_node_id = node_ids[j]
                    results_key = frozenset((i_node_id, j_node_id))
                    if results_key in results or i_node_id == j_node_id:
                        continue

                    j_node = graph.get_node(j_node_id)

                    prediction = scorer_fn(i_node, j_node)
                    label = i_node.connected_to_id(j_node_id)

                    if label == True and prediction == label:
                        true_positive += 1
                    elif label == True and prediction != label:
                        false_negative += 1
                    elif label == False and prediction == label:
                        true_negative += 1
                    elif label == False and prediction != label:
                        false_positive += 1

                    results.add(results_key)
    
            results_queue.put((false_positive, false_negative, true_positive, true_negative))
        except Exception as e:
            print(e)

    range_pairs = get_worker_range_pairs(node_ids, num_workers)

    range_idx = 0
    result_procs = []

    total = 0
    while range_idx < len(range_pairs):
        r1, r2 = range_pairs[range_idx]
        r1_low, r1_high = r1
        r2_low, r2_high = r2

        result_proc = Process(target=eval_helper, args=(graph, node_ids, r1_low, r1_high, r2_low, r2_high, results_queue))
        result_proc.start()
        result_procs.append(result_proc)

        if len(result_procs) >= num_workers:
            for proc in result_procs:
                proc.join()
            result_procs = []

        range_idx += 1

    for proc in result_procs:
        proc.join()

    total_false_positive = 0
    total_false_negative = 0
    total_true_positive = 0
    total_true_negative = 0

    total_explored = 0
    while not results_queue.empty():
        false_positive, false_negative, true_positive, true_negative = results_queue.get()
        total_false_positive += false_positive
        total_false_negative += false_negative
        total_true_positive += true_positive
        total_true_negative += true_negative
        
        total_explored += (false_positive + false_negative + true_positive + true_negative)

    sensitivity = float(total_true_positive) / max(1, (total_true_positive + total_false_negative))
    specificity = float(total_true_negative) / max(1, (total_true_negative + total_false_positive))
    precision = float(total_true_positive) / max(1, (total_true_positive + total_false_positive))
    false_positive_rate = float(total_false_positive) / max(1, (total_false_positive + total_true_negative))

    stats_results = {
                       RESULTS_KEY_SENSITIVITY : sensitivity,
                       RESULTS_KEY_SPECIFICITY : specificity,
                       RESULTS_KEY_PRECISION : precision,
                       RESULTS_KEY_FALSE_POSITIVE_RATE : false_positive_rate,
                       RESULTS_KEY_TOTAL_EXPLORED : total_explored
                    }

    print("Total explored: {}".format(total_explored))
    print("Sensitivity (TPR): {}, Specificity: {}, Precision: {}, FPR: {}".format(sensitivity, specificity, precision, false_positive_rate))

    return stats_results 
