# Create a file with the number of principal components for different values of pca explained variance

from __future__ import print_function

import os
from argparse import ArgumentParser
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    parser = ArgumentParser("")
    parser.add_argument("feats", help="Path to the npy features.")
    parser.add_argument("out_dir", help="Path where to save the processed files.")
    parser.add_argument("prefix", help="Files prefix, name of the dataset.")
    args = parser.parse_args()
    feats_path = args.feats
    out_dir = args.out_dir
    file_prefix = args.prefix

    feats = np.load(feats_path)

    # perform standardization
    feats = StandardScaler().fit_transform(feats)
    results = []

    for pca_variance in [.95,.90,.85,.80,.75,.70,.65,.60,.55,.50,.45,.40,.35,.30,.25,.20,.15,.10,.05]:
        pca = PCA(n_components=pca_variance, svd_solver="full")
        feats_pc = pca.fit_transform(feats)
        results.append((pca_variance, feats_pc.shape[1]))
        print((pca_variance, feats_pc.shape[1]))

    # Create output directory if not present
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(os.path.join(out_dir, file_prefix + '-pca.txt'), 'w+') as fout:
        for result in results:
            fout.write(str(result)+'\n')
