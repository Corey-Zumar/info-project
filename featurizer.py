import sys
import os
import argparse
import numpy as np

EXTENSION_GRAPH_FEAT = ".feat"
EXTENSION_GRAPH_EGOFEAT = ".egofeat"
EXTENSION_GRAPH_FEATNAMES = ".featnames"

def create_feature_vectors(dataset_dir, ego_node_id):
    feature_vecs = {}
    feats_path = os.path.join(dataset_dir, str(ego_node_id) + EXTENSION_GRAPH_FEAT)
    with open(feats_path, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]
        for features_line in lines:
            features = [int(item) for item in features_line.split(" ")]
            node_id, features = features[0], np.array(features[1:], dtype=np.float32)
            feature_vecs[node_id] = features

    ego_feats_path = os.path.join(dataset_dir, str(ego_node_id) + EXTENSION_GRAPH_EGOFEAT)
    with open(ego_feats_path, "r") as f:
        ego_features = f.readline().rstrip()
        ego_features = np.array([int(item) for item in ego_features.split(" ")], dtype=np.float32)
        feature_vecs[ego_node_id] = features

    return feature_vecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create facebook graph feature vectors')
    parser.add_argument('-d', '--dir_path', type=str, help='Path to a directory containing features files')
    parser.add_argument('-e', '--ego_node_id', type=int, help='The id of the ego node from which to derive features for the graph')

    args = parser.parse_args()

    print(create_feature_vectors(args.dir_path))
