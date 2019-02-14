from __future__ import print_function

import sys

import numpy as np
from numpy import random as rnd

BATCH_SIZE=128
IDENTITY_DIM=16
BASE_LOG_DIR="./logs"
VALIDATE_ITER=256
VALIDATE_BATCH_SIZE=-1
GPU=1
PRINT_EVERY=100
SUPERVISED_RATIO=0.5
POS_CLASS=0

if __name__ == "__main__":

    train_prefix = sys.argv[1]
    version = sys.argv[2]
    epochs = sys.argv[3]
    iterations = sys.argv[4]
    out_file = sys.argv[5]

    with open(out_file, "w") as fp:
        for i in range(int(iterations)):
            note=version
            if version=="casage":
                sampler="uniform"
                topology_label_ratio=0
            elif version=="lasage":
                sampler="label_assisted"
                topology_label_ratio=rnd.random_sample()
            else:
                print("unsupported version")
                sys.exit(1)

            models=["graphsage_mean","graphsage_seq","graphsage_maxpool","graphsage_meanpool"]
            model=models[rnd.randint(len(models))]

            model_sizes=["small","big"]
            model_size=model_sizes[rnd.randint(len(model_sizes))]

            dropout=abs(rnd.normal(0.1,0.1))

            learning_rate=abs(rnd.normal(0.15,0.05))

            weight_decay=abs(rnd.normal(0.,0.1))

            max_degrees=[32,64,128,256,516]
            max_degree=max_degrees[rnd.randint(len(max_degrees))]

            samples_1=int(abs(rnd.normal(25,5)))
            samples_2=int(abs(rnd.normal(15,5)))
            samples_3=int(abs(rnd.normal(5,5)))

            dim_1=int(abs(rnd.normal(128,30)))
            dim_2=int(abs(rnd.normal(128,30)))

            neg_sample_size=int(abs(rnd.normal(30,10)))

            string=(
                "./venv/bin/python -m graphsage.semisupervised_train \\\n"
                "--model %s \\\n"
                "--sampler %s \\\n"
                "--train_prefix %s \\\n"
                "--learning_rate %s \\\n"
                "--model_size %s \\\n"
                "--epochs %s \\\n"
                "--dropout %s \\\n"
                "--weight_decay %s \\\n"
                "--max_degree %s \\\n"
                "--samples_1 %s \\\n"
                "--samples_2 %s \\\n"
                "--samples_3 %s \\\n"
                "--dim_1 %s \\\n"
                "--dim_2 %s \\\n"
                "--neg_sample_size %s \\\n"
                "--batch_size %s \\\n"
                "--identity_dim %s \\\n"
                "--base_log_dir %s \\\n"
                "--validate_iter %s \\\n"
                "--validate_batch_size %s \\\n"
                "--gpu %s \\\n"
                "--print_every %s \\\n"
                "--supervised_ratio %s \\\n"
                "--pos_class %s \\\n"
                "--topology_label_ratio %s \\\n"
                "--note %s \n"
                "\n"
            ) % (model, sampler, train_prefix, learning_rate, model_size, epochs, dropout, weight_decay, max_degree, samples_1, samples_2, samples_3, dim_1, dim_2, neg_sample_size, BATCH_SIZE, IDENTITY_DIM, BASE_LOG_DIR, VALIDATE_ITER, VALIDATE_BATCH_SIZE, GPU, PRINT_EVERY, SUPERVISED_RATIO, POS_CLASS, topology_label_ratio, note)

            fp.write(string)
