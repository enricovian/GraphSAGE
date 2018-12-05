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
    parser.add_argument("prefix", help="Files prefix.")
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

    relevant_genes = []
    with open(os.path.join(in_dir, "relevant_genes.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0: # skip the column names
                relevant_genes.append(row[0])
            line_count += 1
        print('{:d} relevant genes.'.format(len(relevant_genes)))

    irrelevant_genes = []
    with open(os.path.join(in_dir, "irrelevant_genes.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            irrelevant_genes.append(row[0])
            line_count += 1
        print('{:d} irrelevant genes.'.format(len(irrelevant_genes)))

    nodes = {}
    with open(os.path.join(in_dir, "hint+hi2012_index_file.txt")) as file:
        for line in file:
            index, node_name, _ = re.split('\t| ', line)
            nodes[int(index)] = node_name
        print('{:d} nodes'.format(len(nodes)))

    edges = []
    with open(os.path.join(in_dir, "hint+hi2012_edge_file.txt")) as file:
        for line in file:
            index1, index2, _ = re.split('\t| ', line)
            edges.append((int(index1), int(index2)))
        print('{:d} edges'.format(len(edges)))

    # Perform node set split
    print("Splitting node set..")
    perm = np.random.permutation(len(nodes)-1)
    train_idx = perm[:(int(len(nodes)*DATA_SPLIT[0]))]
    val_idx = perm[(int(len(nodes)*DATA_SPLIT[0])+1):(int(len(nodes)*(DATA_SPLIT[0]+DATA_SPLIT[1])))]
    test_idx = perm[(int(len(nodes)*(DATA_SPLIT[0]+DATA_SPLIT[1]))+1):]
    print("{:d} train nodes".format(len(train_idx)))
    print("{:d} val nodes".format(len(val_idx)))
    print("{:d} test nodes".format(len(test_idx)))

    # Construct networkx graph
    G = nx.empty_graph()
    labels = {}
    node_integers = {} # integers used to map the nodes to a line in the features file
    n = 0
    for node_id, node_name in nodes.items():
        node_labels = [0,0]
        if node_name in relevant_genes:
            node_labels[0] = 1
        if node_name in irrelevant_genes:
            node_labels[1] = 1
        G.add_node(
            node_id,
            name=node_name,
            val=(n in val_idx),
            test=(n in test_idx))
        labels[node_id] = node_labels
        node_integers[node_id] = n
        n += 1
    for node1, node2 in edges:
        G.add_edge(node1, node2)

    # Dump the data to files
    json_data = json_graph.node_link_data(G)
    with open(os.path.join(out_dir, file_prefix + '-G.json'), 'w+') as file:
        json.dump(json_data, file)
    with open(os.path.join(out_dir, file_prefix + '-class_map.json'), 'w+') as file:
        json.dump(labels, file)
    with open(os.path.join(out_dir, file_prefix + '-id_map.json'), 'w+') as file:
        json.dump(node_integers, file)
