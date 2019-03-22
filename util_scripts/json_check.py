import json
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser("")
    parser.add_argument("json_file", help="Path to the json file.")
    args = parser.parse_args()
    json_file = args.json_file

    with open(json_file) as f:
        data = json.load(f)

    print("Number of items {}".format(len(data)))
