import matplotlib
matplotlib.use('Agg') # use non-interactive backend

import numpy as np
import random
import json
import sys
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# networkx
import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

DATA_SPLIT = [.75,.25] # ratio for train and validation set (the remaining is test set)
N_NODES = 10000 # number of nodes in the new graph
SEED = 10

# perform BFS to reduce reddit dataset
if __name__ == '__main__':
    parser = ArgumentParser("")
    parser.add_argument("in_dir", help="Path to the input file directory.")
    parser.add_argument("out_dir", help="Path where to save the processed files.")
    parser.add_argument("prefix", help="Files prefix, name of the dataset.")
    parser.add_argument("seed", help="Seed for the rng.")
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    file_prefix = args.prefix
    SEED = int(args.seed)

    # set random seed
    np.random.seed(SEED)

    # load original graph
    feats = np.load(os.path.join(in_dir, file_prefix+"-feats.npy"))
    id_map = json.load(open(os.path.join(in_dir, file_prefix+"-id_map.json")))
    id_map = {k:int(v) for k,v in id_map.items()}
    class_map = json.load(open(os.path.join(in_dir, file_prefix+"-class_map.json")))
    class_map = {k:int(v) for k,v in class_map.items()}
    G_data = json.load(open(os.path.join(in_dir, file_prefix+"-G.json")))
    G = json_graph.node_link_graph(G_data)
    print("Originally {} nodes.".format(len(G)))

    # compute reduced graph using bfs
    starting_node = random.choice(G.nodes())
    edges = list(nx.bfs_edges(G,starting_node))
    nodes_reduced = []
    for edge in edges:
        node = edge[0]
        if node not in nodes_reduced:
            nodes_reduced.append(node)
        if len(nodes_reduced) >= N_NODES:
            break
    G_reduced = G.subgraph(nodes_reduced)
    print("The new graph has {} nodes.".format(len(G_reduced)))

    # Perform node set split
    print("Splitting node set..")
    perm = np.random.permutation(len(G_reduced)-1)
    train_idx = perm[:(int(len(G_reduced)*DATA_SPLIT[0]))]
    val_idx = perm[(int(len(G_reduced)*DATA_SPLIT[0])+1):(int(len(G_reduced)*(DATA_SPLIT[0]+DATA_SPLIT[1])))]
    test_idx = perm[(int(len(G_reduced)*(DATA_SPLIT[0]+DATA_SPLIT[1]))+1):]
    print("{:d} train nodes".format(len(train_idx)))
    print("{:d} val nodes".format(len(val_idx)))
    print("{:d} test nodes".format(len(test_idx)))

    # compute auxiliary files for the reduced dataset and mark val and test nodes
    i = 0
    classes = {}
    new_id_map = {}
    new_class_map = {}
    new_features = []
    for node in G_reduced.nodes():
        G_reduced.node[node]['val'] = (i in val_idx)
        G_reduced.node[node]['test'] = (i in test_idx)
        feat = feats[id_map[node]]
        new_features.append(feat)
        new_id_map[node] = i
        new_class_map[node] = class_map[node]
        if class_map[node] in classes:
            classes[class_map[node]] += 1
        else:
            classes[class_map[node]] = 1
        i += 1

    # Create output directory if not present
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # plot classes count
    class_labels = [k for k,v in classes.items()]
    class_counts = [v for k,v in classes.items()]
    plt.bar(class_labels, class_counts, align='center', alpha=0.5)
    plt.ylabel('Count')
    plt.title('Classes count in reduced data')
    plt.savefig(open(os.path.join(out_dir, file_prefix + '-count'+str(SEED)+'.png'), 'w+'))

    # write the reduced dataset to files
    json_data = json_graph.node_link_data(G_reduced)
    with open(os.path.join(out_dir, file_prefix + '-G.json'), 'w+') as file:
        json.dump(json_data, file)
    with open(os.path.join(out_dir, file_prefix + '-class_map.json'), 'w+') as file:
        json.dump(new_class_map, file)
    with open(os.path.join(out_dir, file_prefix + '-id_map.json'), 'w+') as file:
        json.dump(new_id_map, file)
    np.save(os.path.join(out_dir, file_prefix + '-feats.npy'), np.array(new_features))
