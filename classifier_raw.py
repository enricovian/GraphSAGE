import sys
import json
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

TRAIN_RATIO = 0.7

if __name__ == "__main__":
    prefix = sys.argv[1]
    seed = int(sys.argv[2])

    # Set random seed
    np.random.seed(seed)

    label_map = json.load(open(prefix + "-class_map.json"))
    label_map = {int(k):np.argmax(v) for k,v in label_map.items()} # label is a one-hot vector

    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {int(k):int(v) for k,v in id_map.items()}

    feats = np.load(prefix + "-feats.npy")
    feats_label = []
    for id, n in id_map.items():
        feats_label.append((feats[n], label_map[id]))

    random.shuffle(feats_label)
    train = feats_label[:int(len(feats_label)*TRAIN_RATIO)-1]
    test = feats_label[int(len(feats_label)*TRAIN_RATIO)-1:]
    train_feats = []
    test_feats = []
    train_label = []
    test_label = []
    for feat, label in train:
        train_feats.append(feat)
        train_label.append(label)
    for feat, label in test:
        test_feats.append(feat)
        test_label.append(label)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(
        train_feats, train_label
    )
    train_preds = clf.predict(train_feats)
    test_preds = clf.predict(test_feats)

    correct_train = 0
    for i, pred in enumerate(train_preds):
        if train_label[i] == pred:
            correct_train += 1
    accuracy_train = float(correct_train) / float(len(train_preds))

    correct_test = 0
    for i, pred in enumerate(test_preds):
        if test_label[i] == pred:
            correct_test += 1
    accuracy_test = float(correct_test) / float(len(test_preds))

    # print results
    print("Train accuracy: {:.5f}".format(accuracy_train))
    print("Test accuracy: {:.5f}".format(accuracy_test))
