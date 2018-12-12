import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.classifiers import ClassifierInfo, EmbeddingsClassifier
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
from graphsage.prediction import BipartiteEdgePredLayer

flags = tf.app.flags
FLAGS = flags.FLAGS

class SemiSupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean",
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
            - identity_dim: Dimension of identity feature for nodes (optional)
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs = placeholders["batch"]
        self.inputs_pos = placeholders["batch_pos"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.sup_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.unsup_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.optimizer = tf.cond(self.placeholders['supervised'], self.sup_optimizer, self.unsup_optimizer)

        self.build()

    def build(self):
        # initialize negative sampler (?)
        labels = tf.reshape(
            tf.cast(self.inputs_pos, dtype=tf.int64),
            [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        # sample neighbors and perform aggregation (for every layer defined in layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        samples, support_sizes = self.sample(self.inputs, self.layer_infos) # TODO: Look at the sampler
        self.outputs, self.aggregators = self.aggregate(samples, [self.features], self.dims, num_samples,
            support_sizes, concat=self.concat, model_size=self.model_size)
        self.outputs = tf.nn.l2_normalize(self.outputs, 1) # normalization

        samples_pos, support_sizes_pos = self.sample(self.inputs_pos, self.layer_infos)
        self.outputs_pos, _ = self.aggregate(samples_pos, [self.features], self.dims, num_samples,
            support_sizes_pos, aggregators=self.aggregators, concat=self.concat,
            model_size=self.model_size)
        self.outputs_pos = tf.nn.l2_normalize(self.outputs_pos, 1) # normalization

        samples_neg, support_sizes_neg = self.sample(self.neg_samples, self.layer_infos,
            FLAGS.neg_sample_size)
        self.outputs_neg, _ = self.aggregate(samples_neg, [self.features], self.dims, num_samples,
            support_sizes_neg, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
            concat=self.concat, model_size=self.model_size)
        self.outputs_neg = tf.nn.l2_normalize(self.outputs_neg, 1) # normalization

        # TODO: write description
        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
            dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid,
            bilinear_weights=False,
            name='edge_predict')

        # define and apply classifier
        dim_mult = 2 if self.concat else 1
        info = [ClassifierInfo(dim_mult*self.dims[-1]/2, self.placeholders['dropout'], tf.nn.relu)]
        self.classifier = EmbeddingsClassifier(dim_mult*self.dims[-1], self.num_classes, info)
        self.node_preds = self.classifier(self.outputs)

        # compute relevant metrics
        aff = self.link_pred_layer.affinity(self.outputs, self.outputs_pos) # affinity
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs, self.outputs_neg)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        # 'read' vars return the computed value so far without altering the local streaming varialbles
        self.accuracy_read, self.accuracy = self._accuracy(name="train")
        self.accuracy_read_val, self.accuracy_val = self._accuracy(name="val")

        # compute loss
        # self.loss = tf.cond(self.placeholders['supervised'], self._loss_sup, self._loss_unsup)
        self.loss_sup = self._loss_sup()
        self.loss_unsup = self._loss_unsup()

        # supervised gradients and optimization
        sup_grads_and_vars = self.sup_optimizer.compute_gradients(self.loss_sup)
        sup_clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
            for grad, var in sup_grads_and_vars]
        self.sup_grad, _ = sup_clipped_grads_and_vars[0]
        self.sup_opt_op = self.sup_optimizer.apply_gradients(sup_clipped_grads_and_vars)

        # unsupervised gradients and optimization
        unsup_grads_and_vars = self.unsup_optimizer.compute_gradients(self.loss_unsup)
        unsup_clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
        for grad, var in unsup_grads_and_vars]
        self.unsup_grad, _ = unsup_clipped_grads_and_vars[0]
        self.unsup_opt_op = self.unsup_optimizer.apply_gradients(unsup_clipped_grads_and_vars)

        # predict
        self.preds = self.predict()

    def _loss_unsup(self):
        loss = 0
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        loss += self.link_pred_layer.loss(self.outputs, self.outputs_pos, self.outputs_neg)
        loss = loss / tf.cast(self.batch_size, tf.float32)
        return loss

    def _loss_sup(self):
        loss = 0
        # Weight decay loss (penalty on high model parameters)
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.classifier.var_values():
            loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # classification loss (cross entropy)
        if self.sigmoid_loss:
            loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.node_preds,
                labels=self.placeholders['labels']))
        else:
            loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.node_preds,
                labels=self.placeholders['labels']))
        return loss

    def _accuracy(self, name):
        labels = self.placeholders['labels']
        preds = self.node_preds
        # acc, acc_op = tf.metrics.accuracy(labels, preds)
        acc, acc_op = tf.metrics.accuracy(
            labels=tf.argmax(labels, 1),
            predictions=tf.argmax(preds, 1),
            name=name+"_acc")
        return acc, acc_op

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
