import sys
import os
import argparse
import numpy as np

EXTENSION_GRAPH_FEAT = ".feat"

def create_feature_vectors(dataset_dir):
    feature_vecs = {}
    feats_paths = [os.path.join(dataset_dir, fname) for fname in os.listdir(dataset_dir) if fname.endswith(EXTENSION_GRAPH_FEAT)]
    for feats_path in feats_paths:
        with open(feats_path, "r") as f:
            lines = [line.rstrip() for line in f.readlines()]
            for features_line in lines:
                features = [int(item) for item in features_line.split(" ")]
                node_id, features = features[0], np.array(features[1:], dtype=np.uint32)
                feature_vecs[node_id] = features

    return feature_vecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create facebook graph feature vectors')
    parser.add_argument('-d', '--dir_path', type=str, help='Path to a directory containing features files')

    args = parser.parse_args()

    print(create_feature_vectors(args.dir_path))
