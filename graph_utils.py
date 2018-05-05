import sys
import os
import argparse
import itertools
import numpy as np
import math

from datetime import datetime
from threading import RLock
from multiprocessing import Process
from multiprocessing import Queue as MPQueue

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

    def  __init__(self, node_id):
        self.node_id = node_id
        # A list of nodes to which the current
        # node is connected via an undirected edge
        self.connections = set()
        self.connection_ids = set()
        self.node_id = node_id

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

def construct_network_graph(graph_file_path):
    with open(graph_file_path, "r") as f:
        connection_lines = [line.rstrip().split(" ") for line in f.readlines()]

    connections = [(int(source), int(dest)) for source, dest in connection_lines]
    sources, dests = zip(*connections)
    unique_nodes = set(sources + dests)

    network_graph = Graph()

    for node_node_id in unique_nodes:
       new_node = Node(node_node_id)
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

def create_scorer(method_fn, correctness_threshold):
    return lambda node_1, node_2 : method_fn(node_1, node_2) >= correctness_threshold

def eval_link_prediction_method(graph, scorer_fn, num_workers=20):
    node_ids = list(graph.get_node_ids())

    results_queue = MPQueue()

    def eval_helper(graph, node_ids, r1_low, r1_high, r2_low, r2_high, results_queue):
        try:
            print(r1_low, r1_high, r2_low, r2_high)
            results = set()
            correct_negative = 0
            correct_positive = 0
            total_negative = 0
            total_positive = 0
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

                    if label == True:
                        total_positive += 1
                        if prediction == label:
                            correct_positive += 1

                    else:
                        total_negative += 1
                        if prediction == label:
                            correct_negative += 1

            results_queue.put((correct_negative, correct_positive, total_negative, total_positive))
        except Exception as e:
            print(e)

    num_intervals = max(1, int(math.sqrt(num_workers)))
    interval_length = int(len(node_ids) / num_intervals)
    idxs = np.cumsum([0] + [interval_length for _ in range(num_intervals - 1)])
    idxs = np.append(idxs, len(node_ids))
    ranges = []
    for i in range(len(idxs) - 1):
        r_low = idxs[i]
        r_high = idxs[i + 1]
        ranges.append((r_low, r_high))

    range_pairs = [(r1, r2) for r1, r2 in itertools.product(ranges, ranges)]
    unique_set = set()
    new_pairs = []
    for pair in range_pairs:
        unique_key = frozenset(pair)
        if unique_key not in unique_set:
            new_pairs.append(pair)
            unique_set.add(unique_key)

    range_pairs = new_pairs

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

        range_idx += 1

    for proc in result_procs:
        proc.join()

    total_correct_negative = 0
    total_correct_positive = 0
    total_negative = 0
    total_positive = 0
    while not results_queue.empty():
        correct_negative, correct_positive, explored_negative, explored_positive = results_queue.get()
        total_correct_negative += correct_negative
        total_correct_positive += correct_positive
        total_negative += explored_negative
        total_positive += explored_positive

    # total_false_positive = (
    # true_positive_ratio = float(total_correct_positive) / total_positive
    # false_positive_ratio = 
    #
    # print("Total Correct: {}, Total Explored: {}, Ratio: {}".format(total_correct, total_explored, float(total_correct) / total_explored))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct a facebook graph given node and edge inputs')
    parser.add_argument('-p', '--path', type=str, help='Path to the facebook combined graph file')

    args = parser.parse_args()

    before = datetime.now()

    graph = construct_network_graph(args.path)
    # common_neighbors_scorer = create_scorer(method_common_neighbors, 1)
    common_neighbors_scorer = create_scorer(method_common_neighbors_exclude_selves, 10)
    eval_link_prediction_method(graph, common_neighbors_scorer, 64)

    end = datetime.now()
    print((end - before).total_seconds())
