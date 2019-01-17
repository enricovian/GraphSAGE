from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists

class LabelAssistedNeighborSampler(Layer):
    """
    Samples from both structural neighbors and nodes with the same class.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, label_adj_info, topology_label_ratio, **kwargs):
        super(LabelAssistedNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.label_adj_info = label_adj_info
        self.topology_label_ratio = topology_label_ratio

    def _call(self, inputs):
        ids, num_samples = inputs
        num_samples_adj = int(num_samples*self.topology_label_ratio)
        num_samples_label = num_samples - num_samples_adj
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples_adj])
        label_adj_lists = tf.nn.embedding_lookup(self.label_adj_info, ids)
        label_adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(label_adj_lists)))
        label_adj_lists = tf.slice(label_adj_lists, [0,0], [-1, num_samples_label])
        complete_adj_lists = tf.concat([adj_lists, label_adj_lists], axis=1)
        return complete_adj_lists
