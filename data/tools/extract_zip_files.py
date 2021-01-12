# Copyright 2020 Xiaomi
#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import utils


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        required=True,
                        dest='dir',
                        help="Input data files directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if os.path.isdir(args.dir):
        utils.unzip_directory_recursively(args.dir)
    else:
        raise Exception("invalid input file path: {0}.".format(args.dir))
