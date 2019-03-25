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
TRAIN_LABELS = 1 # ratio for training labels to consider, if 1 all labels are kept
SEED = 123

if __name__ == '__main__':
    parser = ArgumentParser("")
    parser.add_argument("in_dir", help="Path to the input file directory.")
    parser.add_argument("out_dir", help="Path where to save the processed files.")
    parser.add_argument("prefix", help="Files prefix, name of the dataset.")
    parser.add_argument("seed", help="Seed for the rng.")
    parser.add_argument("train_labels", help="Ratio for training labels to consider, if 1 all labels are kept.")
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    file_prefix = args.prefix
    SEED = int(args.seed)
    TRAIN_LABELS = float(args.train_labels)

    # Set random seed
    np.random.seed(SEED)

    # Create output directory if not present
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # read nodes data
    ids = []
    features_dict = {}
    labels = {}
    class_count = {}
    with open(os.path.join(in_dir, file_prefix+"-nodes.tab")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        n_row = 0 # first two rows are info
        for row in csv_reader:
            if n_row >= 2: # first two rows are info
                id = int(row[0])
                ids.append(id)
                label = int(re.findall("\d$", row[1])[0])-1
                labels[id] = label
                try:
                    class_count[label] += 1
                except Exception as e:
                    class_count[label] = 1
                features_raw = row[2:-1]
                features = {}
                for feat in features_raw:
                    split = feat.split("=")
                    feat_name = split[0]
                    feat_value = float(split[1])
                    features[feat_name] = feat_value
                features_dict[id] = features
            elif n_row == 1: # second row contains features names
                titles = row[1:-1] # first and last values are not relevant
                titles = [t.split(":")[1] for t in titles]
            n_row += 1
    print("Number of nodes: {}".format(len(ids)))
    print("Number of features: {}".format(len(titles)))
    print("Classes count: {}".format(class_count))

    # set to 0. features not present and convert the features to a list
    for id in features_dict:
        features = features_dict[id]
        features_list = []
        for title in titles:
            if title not in features:
                features[title] = 0.
            features_list.append(features[title])
        features_dict[id] = features_list

    # read edges data
    cites = []
    with open(os.path.join(in_dir, file_prefix+"-cites.tab")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        n_row = 0 # first two rows are info
        for row in csv_reader:
            if n_row >= 2: # first two rows are info
                first = int(row[1].split(":")[1])
                second = int(row[3].split(":")[1])
                cites.append((first,second))
            n_row += 1
    print("Number of edges: {}".format(len(cites)))

    # convert labels to one hot vectors
    oh_labels = {}
    for id in labels:
        oh_labels[id] = np.eye(len(class_count), dtype=int)[labels[id]].tolist()

    # Perform node set split
    print("Splitting node set..")
    perm = np.random.permutation(ids)
    train_idx = perm[:(int(len(ids)*DATA_SPLIT[0]))]
    train_perm = np.random.permutation(train_idx)
    unlabeled_train_idx = train_perm[:(int(len(train_perm)*(1.-TRAIN_LABELS)))]
    val_idx = perm[(int(len(ids)*DATA_SPLIT[0])+1):(int(len(ids)*(DATA_SPLIT[0]+DATA_SPLIT[1])))]
    test_idx = perm[(int(len(ids)*(DATA_SPLIT[0]+DATA_SPLIT[1]))+1):]
    print("{:d} train nodes".format(len(train_idx)))
    print("{:d} val nodes".format(len(val_idx)))
    print("{:d} test nodes".format(len(test_idx)))

    # remove train labels according to the TRAIN_LABELS flag
    removed=0
    for id in ids:
        if id in unlabeled_train_idx:
            oh_labels[id] = np.zeros(len(oh_labels[id]), int).tolist()
            removed+=1
    print("Removed {:d} training labels. Training set is {:.2f}% labeled".format(removed,100*TRAIN_LABELS))

    # construct networkx graph
    G = nx.empty_graph()
    node_integers = {}
    for id in ids:
        G.add_node(
            id,
            name=id,
            val=(id in val_idx),
            test=(id in test_idx))
    for node1, node2 in cites:
        G.add_edge(node1, node2)

    # convert features dict to a list
    id_map = {} # map associating ids to order in the feature list
    features_list = []
    n = 0
    for id in ids:
        if id in features_dict:
            features_list.append(features_dict[id])
            id_map[id] = n
            n += 1

    # Dump the data to files
    json_data = json_graph.node_link_data(G)
    with open(os.path.join(out_dir, file_prefix + '-G.json'), 'w+') as file:
        json.dump(json_data, file)
    with open(os.path.join(out_dir, file_prefix + '-class_map.json'), 'w+') as file:
        json.dump(oh_labels, file)
    with open(os.path.join(out_dir, file_prefix + '-id_map.json'), 'w+') as file:
        json.dump(id_map, file)
    np.save(os.path.join(out_dir, file_prefix + '-feats.npy'), np.array(features_list))
