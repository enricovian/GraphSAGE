import sys
import json
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

TRAIN_RATIO = 0.7
NUM_CLASSES = 3

if __name__ == "__main__":
    embeds_dir = sys.argv[1]
    prefix = sys.argv[2]

    label_map = json.load(open(prefix + "-class_map.json"))
    label_map = {int(k):np.argmax(v) for k,v in label_map.items()} # label is a one-hot vector

    embeds = np.load(embeds_dir + "val.npy")
    with open(os.path.join(embeds_dir, "val.txt")) as f:
        ids = f.read().splitlines()
        ids = [int(id) for id in ids]
    embeds_map = {id:embeds[n] for n,id in enumerate(ids)}
    embeds_label = [(embed,label_map[id]) for id,embed in embeds_map.items()]
    random.shuffle(embeds_label)
    train = embeds_label[:int(len(embeds_label)*TRAIN_RATIO)-1]
    test = embeds_label[int(len(embeds_label)*TRAIN_RATIO)-1:]
    train_embeds = []
    test_embeds = []
    train_label = []
    test_label = []
    for embed, label in train:
        train_embeds.append(embed)
        train_label.append(label)
    for embed, label in test:
        test_embeds.append(embed)
        test_label.append(label)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(
        train_embeds, train_label
    )
    train_preds = clf.predict(train_embeds)
    test_preds = clf.predict(test_embeds)

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

    correct_train = 0
    random_train = np.random.choice(range(NUM_CLASSES), len(train_preds))
    for i, pred in enumerate(random_train):
        if train_label[i] == pred:
            correct_train += 1
    rnd_accuracy_train = float(correct_train) / float(len(random_train))

    correct_test = 0
    rnd_accuracy_test = float(correct_test) / float(len(test_preds))
    random_test = np.random.choice(range(NUM_CLASSES), len(test_preds))
    for i, pred in enumerate(random_test):
        if test_label[i] == pred:
            correct_test += 1
    rnd_accuracy_test = float(correct_test) / float(len(random_test))

    # print results
    print("Train accuracy: {:.5f} - random: {:.5f}".format(accuracy_train, rnd_accuracy_train))
    print("Test accuracy: {:.5f} - random: {:.5f}".format(accuracy_test, rnd_accuracy_test))

    # with open(out_file, "w") as fp:
    #     fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
