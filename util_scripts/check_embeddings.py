import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser("")
    parser.add_argument("npy_path", help="Path to the npy file.")
    args = parser.parse_args()
    npy_path = args.npy_path

    npy = np.load(npy_path)

    for i in range(5):
        print(npy[i])

    print(npy.shape)
