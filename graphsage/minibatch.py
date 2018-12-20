from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

class EdgeMinibatchIterator(object):

    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, id2idx,
            placeholders, context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()
        if context_pairs is None:
            edges = G.edges()
        else:
            edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        if not n2v_retrain:
            self.train_edges = self._remove_isolated(self.train_edges)
            self.val_edges = [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]
        else:
            if fixed_n2v:
                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
            else:
                self.train_edges = self.val_edges = self.edges

        print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
        print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')
        self.val_set_size = len(self.val_edges)

    def _n2v_prune(self, edges):
        is_val = lambda n : self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]

    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        print("Unexpected missing:", missing)
        return new_edge_list

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size,
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test']
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

class NodeMinibatchIterator(object):

    """
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx,
            placeholders, label_map, num_classes,
            batch_size=100, max_degree=25,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]

        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size,
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

class SupervisedEdgeMinibatchIterator(object):

    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    NB: the functions without suffix '_sup' or '_unsup' consider all nodes
    regardless of the labelling (eventually returning an all-0 class list for
    unlabeled entries).
    Instead '_sup' methods consider exclusively labeled nodes and '_unsup'
    methods exclusively unlabeled ones.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, id2idx,
            placeholders, label_map, num_classes, context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.batch_num_sup = 0
        self.batch_num_unsup = 0
        self.label_map = label_map
        self.num_classes = num_classes
        self.labeled_nodes = [n for n in G.nodes() if G.node[n]['labeled']]
        self.unlabeled_nodes = [n for n in G.nodes() if not G.node[n]['labeled']]

        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        train_nodes = [n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]
        test_nodes = [n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]

        if context_pairs is None:
            G_train = G.subgraph(train_nodes)
            train_edges = [e for e in G_train.edges()]
        else:
            train_edges = context_pairs
        self.train_edges = np.random.permutation(train_edges)
        self.train_edges, missing = self._remove_isolated(self.train_edges) # remove edges referring to missing nodes
        print("Unexpected missing nodes:", missing)
        self.train_edges_sup = [edge for edge in self.train_edges if G.node[edge[0]]['labeled']]
        self.train_edges_unsup = [edge for edge in self.train_edges if not G.node[edge[0]]['labeled']]
        self.val_edges = [e for e in G.edges() if e[0] in test_nodes]
        # Put the validation nodes always as first element (DOES MESS UP DIRECTED GRAPHS!)
        self.val_edges.extend([(e[1], e[0]) for e in G.edges() if e[1] in test_nodes])
        self.val_edges_sup = [edge for edge in self.val_edges if G.node[edge[0]]['labeled']]
        self.val_edges_unsup = [edge for edge in self.val_edges if not G.node[edge[0]]['labeled']]
        self.val_set_size = len(self.val_edges)

        print(len(self.train_edges),'train edges -',len(self.train_edges_sup),
            'supervised and',len(self.train_edges_unsup),'unsupervised')
        print(len(self.val_edges),'validation edges -',len(self.val_edges_sup),
            'supervised and',len(self.val_edges_unsup),'unsupervised')
        print(len(train_nodes), 'train nodes -',
            len([n for n in train_nodes if G.node[n]['labeled']]), 'labeled and',
            len([n for n in train_nodes if not G.node[n]['labeled']]), 'unlabeled')
        print(len(test_nodes), 'test nodes -',
            len([n for n in test_nodes if G.node[n]['labeled']]), 'labeled and',
            len([n for n in test_nodes if not G.node[n]['labeled']]), 'unlabeled')

    def _n2v_prune(self, edges):
        is_val = lambda n : self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]

    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        return new_edge_list, missing

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def end_sup(self):
        return self.batch_num_sup * self.batch_size >= len(self.train_edges_sup)

    def end_unsup(self):
        return self.batch_num_unsup * self.batch_size >= len(self.train_edges_unsup)

    def batch_feed_dict(self, batch_edges, duplicates=True):
        """ Construct the feed_dict for a batch of edges.
            The duplicate flag determines whether to consider the same node more than once.
        """
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
        if not duplicates:
            # remove duplicate nodes and update the seen nodes list
            nodes, unique_idx = np.unique(batch1, return_index=True) # remove duplicates
            nodes_unique_idx = [(n, i) for (n, i) in zip(nodes, unique_idx) if n not in self.seen_nodes] # remove nodes seen in previous batches
            if len(nodes_unique_idx) == 0:
                # if there are no new nodes, return None
                return None, None
            nodes, unique_idx = zip(*nodes_unique_idx) # unzip the tuples to lists
            self.seen_nodes.extend(nodes)
            batch1 = [n for i, n in enumerate(batch1) if i in unique_idx]
            batch2 = [n for i, n in enumerate(batch2) if i in unique_idx]
            batch_edges = [n for i, n in enumerate(batch_edges) if i in unique_idx]
        labels = np.vstack([self._make_label_vec(node1) for node1, node2 in batch_edges])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['batch_pos']: batch2})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def next_minibatch_feed_dict_sup(self):
        start_idx = self.batch_num_sup * self.batch_size
        self.batch_num_sup += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges_sup))
        batch_edges = self.train_edges_sup[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def next_minibatch_feed_dict_unsup(self):
        start_idx = self.batch_num_unsup * self.batch_size
        self.batch_num_unsup += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges_unsup))
        batch_edges = self.train_edges_unsup[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def val_feed_dict_sup(self, size=None):
        edge_list = self.val_edges_sup
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def val_feed_dict_unsup(self, size=None):
        edge_list = self.val_edges_unsup
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size,
            len(edge_list))]
        feed_dict, labels = self.batch_feed_dict(val_edges)
        return feed_dict, labels, (iter_num+1)*size >= len(self.val_edges), val_edges


    def incremental_val_feed_dict_sup(self, size, iter_num, duplicates=True):
        """
            The duplicate flag determines whether to consider the same node more than once.
        """
        if iter_num == 0:
            self.seen_nodes = []
        edge_list = self.val_edges_sup
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size,
            len(edge_list))]
        feed_dict, labels = self.batch_feed_dict(val_edges, duplicates=duplicates)
        return feed_dict, labels, (iter_num+1)*size >= len(self.val_edges_sup), val_edges

    def incremental_val_feed_dict_unsup(self, size, iter_num):
        edge_list = self.val_edges_unsup
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size,
            len(edge_list))]
        feed_dict, labels = self.batch_feed_dict(val_edges)
        return feed_dict, labels, (iter_num+1)*size >= len(self.val_edges_unsup), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        if size < 0: # a negative size means the whole set is processed at once
            size = len(node_list)
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        feed_dict, labels = self.batch_feed_dict(val_edges)
        return feed_dict, labels, (iter_num+1)*size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test']
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.train_edges_sup = np.random.permutation(self.train_edges_sup)
        self.train_edges_unsup = np.random.permutation(self.train_edges_unsup)
        # self.val_edges = np.random.permutation(self.val_edges)
        # self.val_edges_sup = np.random.permutation(self.val_edges_sup)
        # self.val_edges_unsup = np.random.permutation(self.val_edges_unsup)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = self.batch_num_sup = self.batch_num_unsup = 0
