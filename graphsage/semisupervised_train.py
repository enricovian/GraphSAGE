from __future__ import division
from __future__ import print_function

import os
import time
from termcolor import colored
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from graphsage.utils import load_data
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.minibatch import SupervisedEdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler, LabelAssistedNeighborSampler
from graphsage.models import SAGEInfo
from graphsage.semisupervised_models import SemiSupervisedGraphsage
from graphsage.formatter import PartialFormatter



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')
flags.DEFINE_string('sampler', 'uniform', 'sampler to be used. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', False, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_float('supervised_ratio', 0.5, 'Probability to perform a supervised training iteration instead of an unsupervised one.')
flags.DEFINE_integer('pos_class', 1, 'Class number to be considered as positive when computing evaluation metrics (e.g precision).')
flags.DEFINE_float('topology_label_ratio', 0.5, 'ratio of topological neighbors and nodes sharing the same class (used for label_assisted sampler).')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_boolean('complete_val', True, "if true the validation graph will contain train nodes as well.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
flags.DEFINE_boolean('print_confusion', False, 'print confusion matrix for supervised iterations and validation')
flags.DEFINE_string('note', '', 'Optional experiment note to append to log directory')

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

placeholders = 0

def get_log_dir():
    timestr = time.strftime("%y%m%d-%H%M%S")
    log_dir = FLAGS.base_log_dir + "/semisup-" + FLAGS.train_prefix.split("/")[-1]
    log_dir += "/{timestamp:s}_{model:s}_{model_size:s}_{lr:0.6f}_({note:s})/".format(
            timestamp=timestr,
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate,
            note=FLAGS.note)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def print_iter(type, epoch, iter, total_steps, loss_sup=None, loss_unsup=None, mrr=None, f1=None, accuracy=None, confusion=None):
    fmt=PartialFormatter(missing='-', missing_spec="7s")
    data = {
        'type': type,
        'epoch': epoch,
        'iter': iter,
        'total_steps': total_steps,
        'loss_sup': loss_sup,
        'loss_unsup': loss_unsup,
        'mrr': mrr,
        'f1': f1,
        'accuracy': accuracy,
        'time': time.strftime("%y.%m.%d-%H:%M:%S")
    }
    line = fmt.format("""[{type:3.3s} {epoch:02d}/{iter:04d} - {total_steps:05d}] [{time:s}] \
loss-sup: {loss_sup:07.4f} loss-unsup: {loss_unsup:07.4f} mrr: {mrr:07.4f} f1-score: {f1:07.4f} \
accuracy: {accuracy:07.4f}""", **data)
    if type == "VAL":
        color = "red"
    elif type == "SUP":
        color = "yellow"
    elif type == "UNS":
        color = "green"
    else:
        color = "grey"
    print(colored(line, color))
    if (confusion is not None and FLAGS.print_confusion):
        print(colored(confusion, color))

# not used
def calc_f1(y_true, y_pred, ignore_missing=False):
    """ Compute f1 scores
        The ignore_missing option specifies whether to ignore labels not present in
        the predictions when computing the metrics.
    """
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    micro_f1 = metrics.f1_score(y_true, y_pred, average="micro",
        labels=(np.unique(y_pred) if ignore_missing else None))
    macro_f1 = metrics.f1_score(y_true, y_pred, average="macro",
        labels=(np.unique(y_pred) if ignore_missing else None))
    return micro_f1, macro_f1

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None, supervised=False):
    t_test = time.time()
    loss = model.loss_sup if supervised else model.loss_unsup
    feed_dict_val, labels = (minibatch_iter.val_feed_dict_sup(size) if supervised else
        minibatch_iter.val_feed_dict(size))
    ops = [loss, model.mrr]
    if supervised:
        feed_dict_val.update({placeholders['pos_class']: FLAGS.pos_class})
        ops.extend([model.accuracy_val, model.f1_val, model.confusion_val, model.preds, model.confusion_val])
        loss_sup, mrr, acc, f1, conf, preds, confusion = sess.run(ops, feed_dict=feed_dict_val)
        loss_unsup = None
    else:
        loss_unsup, mrr = sess.run(ops, feed_dict=feed_dict_val)
        acc = f1 = loss_sup = confusion = None
    return loss_sup, loss_unsup, mrr, acc, f1, confusion, (time.time() - t_test)

# evaluate the whole validation set
def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    sess.run(tf.initializers.variables(tf.local_variables(scope="val"))) # init local variables
    finished = False
    val_losses_sup = []
    val_losses_unsup = []
    val_mrrs = []
    iter_num = 0
    accuracy = None
    f1 = None
    confusion = None

    while not finished:
        feed_dict_val, labels, finished, _ = minibatch_iter.incremental_val_feed_dict_sup(size, iter_num, duplicates=False)
        iter_num += 1
        if feed_dict_val is None:
            continue
        feed_dict_val.update({placeholders['pos_class']: FLAGS.pos_class})
        loss_sup, accuracy, f1, preds, confusion = sess.run(
            [model.loss_sup, model.accuracy_val, model.f1_val, model.preds, model.confusion_val], feed_dict=feed_dict_val)
        val_losses_sup.append(loss_sup)
    finished = False
    while not finished:
        feed_dict_val, _, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        loss_unsup, mrr = sess.run([model.loss_unsup, model.mrr],
                            feed_dict=feed_dict_val)
        val_losses_unsup.append(loss_unsup)
        val_mrrs.append(mrr)
    loss_sup = np.mean(val_losses_sup)
    loss_unsup = np.mean(val_losses_unsup)
    mrr = np.mean(val_mrrs)
    return loss_sup, loss_unsup, mrr, accuracy, f1, confusion, (time.time() - t_test)

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch'),
        'batch_pos' : tf.placeholder(tf.int32, shape=(None), name='batch_pos'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
        'supervised' : tf.placeholder(tf.bool, name='supervised'),
        'pos_class' : tf.placeholder_with_default(1, shape=(), name='pos_class')
    }
    return placeholders

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, _, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run(model.outputs, feed_dict=feed_dict_val)
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str,nodes)))

def train(train_data, test_data=None):

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    global placeholders
    placeholders = construct_placeholders(num_classes)

    # contruct both supervised and unsupervised minibatch iterators
    minibatch = SupervisedEdgeMinibatchIterator(G,
            id_map,
            placeholders,
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree,
            context_pairs=context_pairs,
            complete_validation=FLAGS.complete_val)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    label_adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.label_adj.shape)
    label_adj_info = tf.Variable(label_adj_info_ph, trainable=False, name="label_adj_info")


    # Neighbors sampler
    if FLAGS.sampler == 'uniform':
        sampler = UniformNeighborSampler(adj_info)
    elif FLAGS.sampler == 'label_assisted':
        sampler = LabelAssistedNeighborSampler(adj_info, label_adj_info, FLAGS.topology_label_ratio)
    else:
        raise Exception('Error: sampler name unrecognized.')

    if FLAGS.model == 'graphsage_mean':
        # Layers definitions
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                           SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
        # Create model
        model = SemiSupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos,
            aggregator_type="mean",
            model_size=FLAGS.model_size,
            sigmoid_loss = FLAGS.sigmoid,
            identity_dim = FLAGS.identity_dim,
            logging=True
        )
    elif FLAGS.model == 'gcn':
        # Layers definitions
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]
        # Create model
        model = SemiSupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos,
            aggregator_type="gcn",
            model_size=FLAGS.model_size,
            sigmoid_loss = FLAGS.sigmoid,
            identity_dim = FLAGS.identity_dim,
            concat = False,
            logging=True
        )
    elif FLAGS.model == 'graphsage_seq':
        # Layers definitions
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        # Create model
        model = SemiSupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos,
            aggregator_type="seq",
            model_size=FLAGS.model_size,
            sigmoid_loss = FLAGS.sigmoid,
            identity_dim = FLAGS.identity_dim,
            logging=True
        )
    elif FLAGS.model == 'graphsage_maxpool':
        # Layers definitions
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        # Create model
        model = SemiSupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos,
            aggregator_type="maxpool",
            model_size=FLAGS.model_size,
            sigmoid_loss = FLAGS.sigmoid,
            identity_dim = FLAGS.identity_dim,
            logging=True
        )
    elif FLAGS.model == 'graphsage_meanpool':
        # Layers definitions
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        # Create model
        model = SemiSupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos,
            aggregator_type="meanpool",
            model_size=FLAGS.model_size,
            sigmoid_loss = FLAGS.sigmoid,
            identity_dim = FLAGS.identity_dim,
            logging=True
        )
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    log_dir = get_log_dir()
    sess = tf.Session(config=config)

    val_loss_sup = tf.Variable(0., trainable=False, name="val_loss_sup")
    val_loss_unsup = tf.Variable(0., trainable=False, name="val_loss_unsup")
    val_mrr_var = tf.Variable(0., trainable=False, name="val_mrr")
    with tf.name_scope("train"):
        summary_train_loss_sup = tf.summary.scalar('supervised loss', model.loss_sup)
        summary_train_loss_unsup = tf.summary.scalar('unsupervised loss', model.loss_unsup)
        summary_train_mrr = tf.summary.scalar('mrr', model.mrr)
        summary_train_acc = tf.summary.scalar('accuracy', model.accuracy)
        summary_train_f1 = tf.summary.scalar('f1 score', model.f1)
        summary_train_confusion = tf.summary.image('confusion',
            tf.reshape(tf.cast(model.confusion_read,tf.float32), [1,num_classes,num_classes,1]))
        summary_train_sup = tf.summary.merge([
            summary_train_loss_sup,
            summary_train_mrr,
            summary_train_acc,
            summary_train_f1,
            summary_train_confusion])
        summary_train_unsup = tf.summary.merge([
            summary_train_loss_unsup,
            summary_train_mrr])
    with tf.name_scope("val"):
        summary_val_loss_sup = tf.summary.scalar('supervised loss', val_loss_sup)
        summary_val_loss_unsup = tf.summary.scalar('unsupervised loss', val_loss_unsup)
        summary_val_mrr = tf.summary.scalar('mrr', val_mrr_var)
        summary_val_acc = tf.summary.scalar('accuracy', model.accuracy_read_val) # only read the already computed validation accuracy
        summary_val_f1 = tf.summary.scalar('f1 score', model.f1_read_val) # only read the already computed validation f1 score
        summary_val_confusion = tf.summary.image('confusion',
            tf.reshape(tf.cast(model.confusion_read_val,tf.float32), [1,num_classes,num_classes,1]))
        summary_val = tf.summary.merge([
            summary_val_loss_sup,
            summary_val_loss_unsup,
            summary_val_mrr,
            summary_val_acc,
            summary_val_f1,
            summary_val_confusion])
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer(),
        feed_dict={adj_info_ph: minibatch.adj, label_adj_info_ph: minibatch.label_adj})

    # Train model

    total_steps = 0
    avg_time = 0.0

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    train_label_adj_info = tf.assign(label_adj_info, minibatch.label_adj)
    val_label_adj_info = tf.assign(label_adj_info, minibatch.test_label_adj)
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle() # shuffle the minibatches

        # init local variables
        sess.run(tf.initializers.variables(tf.local_variables(scope="train")))
        sess.run(tf.initializers.variables(tf.local_variables(scope="val")))

        iter = 0
        print('Epoch: %04d' % (epoch + 1))

        while not (minibatch.end() and (minibatch.end_sup() or FLAGS.supervised_ratio==0)):
            # define supervised or unsupervised training
            supervised = False
            if (minibatch.end() or (not minibatch.end_sup() and (np.random.rand() < FLAGS.supervised_ratio))):
                supervised = True
            if supervised:
                loss = model.loss_sup
                optimizer = model.sup_opt_op
            else:
                loss = model.loss_unsup
                optimizer = model.unsup_opt_op
            # Construct feed dictionary
            feed_dict, labels = (minibatch.next_minibatch_feed_dict_sup() if supervised else
                minibatch.next_minibatch_feed_dict())
            feed_dict.update({placeholders['dropout']: FLAGS.dropout}) # TEMP: change placeholder
            feed_dict.update({placeholders['pos_class']: FLAGS.pos_class})

            # Training step
            t = time.time()
            if supervised:
                summary, _, train_cost, train_mrr, preds, confusion = sess.run([
                    summary_train_sup,
                    optimizer, # otimization operation
                    loss, # compute current loss
                    model.mrr,
                    model.preds, # compute predictions for inputs
                    model.confusion],
                    feed_dict=feed_dict
                )
            else:
                summary, _, train_cost, train_mrr, preds, confusion = sess.run([
                    summary_train_unsup,
                    optimizer, # otimization operation
                    loss, # compute current loss
                    model.mrr,
                    model.preds, # compute predictions for inputs
                    model.confusion_read], # only read confusion matrix for unsupervised iterations
                    feed_dict=feed_dict
                )

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run([val_adj_info.op, val_label_adj_info.op])
                if FLAGS.validate_batch_size == -1:
                    val_cost_sup, val_cost_unsup, val_mrr, val_acc, val_f1, val_confusion, duration = incremental_evaluate(
                        sess,
                        model,
                        minibatch,
                        FLAGS.batch_size)
                else:
                    val_cost_sup, val_cost_unsup, val_mrr, val_acc, val_f1, val_confusion, duration = evaluate(
                        sess,
                        model,
                        minibatch,
                        FLAGS.validate_batch_size,
                        supervised=supervised)

                # log validation summary
                if val_cost_sup is not None:
                    sess.run(val_loss_sup.assign(val_cost_sup))
                if val_cost_unsup is not None:
                    sess.run(val_loss_unsup.assign(val_cost_unsup))
                if val_cost_unsup is not None:
                    sess.run(val_mrr_var.assign(val_mrr))
                summary_val_out = sess.run(summary_val, feed_dict=feed_dict)
                summary_writer.add_summary(summary_val_out, total_steps)

                # print validation stats
                print_iter(
                    type="VAL",
                    epoch=epoch+1,
                    iter=iter,
                    total_steps=total_steps,
                    loss_sup=val_cost_sup,
                    loss_unsup=val_cost_unsup,
                    mrr=val_mrr,
                    f1=val_f1,
                    accuracy=val_acc,
                    confusion=val_confusion)

                sess.run([train_adj_info.op, train_label_adj_info.op])

            # log train summary
            summary_writer.add_summary(summary, total_steps)

            # running average for training time
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            # Print training iteration results
            if total_steps % FLAGS.print_every == 0:
                print_iter(
                    type=("SUP" if supervised else "UNS"),
                    epoch=epoch+1,
                    iter=iter,
                    total_steps=total_steps,
                    loss_sup=(train_cost if supervised else None),
                    loss_unsup=(None if supervised else train_cost),
                    mrr=train_mrr,
                    f1=(sess.run(model.f1_read, feed_dict=feed_dict) if supervised else None),
                    accuracy=(sess.run(model.accuracy_read, feed_dict=feed_dict) if supervised else None),
                    confusion=(confusion if supervised else None))

            # update counters
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break

    print("Optimization Finished!")

    # compute final validation results
    sess.run([val_adj_info.op, val_label_adj_info.op])
    val_cost_sup, val_cost_unsup, val_mrr, val_acc, val_f1, val_confusion, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)

    # log final results
    if val_cost_sup is not None:
        sess.run(val_loss_sup.assign(val_cost_sup))
    if val_cost_unsup is not None:
        sess.run(val_loss_unsup.assign(val_cost_unsup))
    if val_cost_unsup is not None:
        sess.run(val_mrr_var.assign(val_mrr))
    summary_val_out = sess.run(summary_val, feed_dict=feed_dict)
    summary_writer.add_summary(summary_val_out, total_steps)

    # print final results
    print("Full validation stats:\n",
          "\tsupervised loss=", "{:.5f}".format(val_cost_sup), "\n",
          "\tunsupervised loss=", "{:.5f}".format(val_cost_unsup), "\n",
          "\tmrr=", "{:.5f}".format(val_mrr), "\n",
          "\taccuracy=", "{:.5f}".format(val_acc), "\n",
          "\tf1-score=", "{:.5f}".format(val_f1), "\n",
          "\tevaluation time=", "{:.5f}".format(duration))
    if FLAGS.print_confusion:
        print("confusion= \n{:s}".format(val_confusion))
    # write an output file
    with open(log_dir + "val_stats.txt", "w") as fp:
        fp.write("supervised_loss={:.5f}, unsupervised_loss={:.5f}, mrr={:.5f}, accuracy={:.5f}, f1-score={:.5f}, evaluation_time={:.5f}".
            format(val_cost_sup, val_cost_unsup, val_mrr, val_acc, val_f1, duration))

    with open(log_dir + "command.txt", "w") as fp:
        fp.write(str(FLAGS.flag_values_dict()))

    # TODO: Perform evaluation on test set

    if FLAGS.save_embeddings:
        print("Saving embeddings..")
        sess.run([val_adj_info.op, val_label_adj_info.op])
        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir)

def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    train(train_data)

if __name__ == '__main__':
    tf.app.run()
