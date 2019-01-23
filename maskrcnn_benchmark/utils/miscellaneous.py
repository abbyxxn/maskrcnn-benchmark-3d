# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import socket
from datetime import datetime


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()


def get_output_dir(output_dir, args, run_name):
    """ Get root output directory for each run """
    cfg_filename, _ = os.path.splitext(os.path.split(args.config_file)[1])
    return os.path.join(output_dir, cfg_filename, run_name)


def get_tensorboard_writer(local_rank, distributed, output_dir):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise ImportError(
            'To use tensorboard please install tensorboardX '
            '[ pip install tensorflow tensorboardX ].'
        )

    if not distributed or (distributed and local_rank == 0):
        summary_logger = SummaryWriter(output_dir)
        return summary_logger
    else:
        return None
