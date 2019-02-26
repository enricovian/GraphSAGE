from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
from argparse import ArgumentParser
import csv
import re

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

DATA_SPLIT = [.75,.25] # ratio for train and validation set (the remaining is test set)
SEED = 123


if __name__ == '__main__':
    parser = ArgumentParser("")
    parser.add_argument("in_dir", help="Path to the input file directory.")
    parser.add_argument("out_dir", help="Path where to save the processed files.")
    parser.add_argument("prefix", help="Files prefix, name of the dataset.")
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    file_prefix = args.prefix

    # Set random seed
    np.random.seed(SEED)

    # Create output directory if not present
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    id_map = {}
    features = []
    labels = {}

    class_map = {}
    class_count = {}
    lastclass = 0

    with open(os.path.join(in_dir, file_prefix+".content")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        id = 0
        for row in csv_reader:
            name = row[0]
            feature = [float(i) for i in row[1:-1]]
            classname = row[-1]
            if classname not in class_map:
                class_map[classname] = lastclass
                lastclass += 1
            label=class_map[classname]
            try:
                class_count[label] += 1
            except Exception as e:
                class_count[label] = 1

            id_map[name]=id
            features.append(feature)
            labels[id]=label
            id += 1
        print('{:d} nodes.'.format(len(id_map)))

    # print(class_map)
    # print(class_count)

    # Convert labels to one hot vectors
    oh_labels = {}
    for id in labels:
        oh_labels[id] = np.eye(len(class_map), dtype=int)[labels[id]].tolist()

    edges = []
    missing = 0
    loops = 0
    with open(os.path.join(in_dir, file_prefix+".cites")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if row[0]==row[1]:
                loops += 1
                continue
            try:
                edges.append((id_map[row[0]], id_map[row[1]]))
            except KeyError as e:
                missing+=1
        print('{:d} edges. {:d} references to missing nodes. {:d} loops.'.format(len(edges), missing, loops))

    # Perform node set split
    print("Splitting node set..")
    perm = np.random.permutation(range(id))
    train_idx = perm[:(int(id*DATA_SPLIT[0]))]
    val_idx = perm[(int(id*DATA_SPLIT[0])+1):(int(id*(DATA_SPLIT[0]+DATA_SPLIT[1])))]
    test_idx = perm[(int(id*(DATA_SPLIT[0]+DATA_SPLIT[1]))+1):]
    print("{:d} train nodes".format(len(train_idx)))
    print("{:d} val nodes".format(len(val_idx)))
    print("{:d} test nodes".format(len(test_idx)))

    # Construct networkx graph
    G = nx.empty_graph()
    n = 0
    node_integers = {}
    for node_name, node_id in id_map.items():
        node_integers[node_id] = node_id
        G.add_node(
            node_id,
            name=node_name,
            val=(node_id in val_idx),
            test=(node_id in test_idx))
    for node1, node2 in edges:
        G.add_edge(node1, node2)

    # Dump the data to files
    json_data = json_graph.node_link_data(G)
    with open(os.path.join(out_dir, file_prefix + '-G.json'), 'w+') as file:
        json.dump(json_data, file)
    with open(os.path.join(out_dir, file_prefix + '-class_map.json'), 'w+') as file:
        json.dump(oh_labels, file)
    with open(os.path.join(out_dir, file_prefix + '-id_map.json'), 'w+') as file:
        json.dump(node_integers, file)
    np.save(os.path.join(out_dir, file_prefix + '-feats.npy'), np.array(features))
