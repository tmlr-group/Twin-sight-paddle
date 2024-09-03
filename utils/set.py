import os
import sys
import time
import os.path as osp
import random
# import paddle
import paddle
import numpy as np
from paddle.optimizer import Optimizer
import math
import paddle.nn.functional as F

__all__ = ['setup_run', 'Logger',  'setup_logger', 'set_random_seed', 'accuracy', 'AverageMeter','ranges']

ranges=[151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173
    , 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196,
       197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
       220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
       243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
       266, 267, 268,281, 282, 283, 284, 285,32, 30, 31,33, 34, 35, 36, 37,80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
       91, 92, 93, 94, 95, 96, 97, 98, 99, 100,365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379,
       380, 381, 382,389, 390, 391, 392, 393, 394, 395, 396, 397,120, 121, 118, 119,300, 301, 302, 303, 304, 305, 306,
       307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319]

def subclass_label_mapping(classes, class_to_idx, ranges):
    # add wildcard
    # range_sets.append(set(range(0, 1002)))
    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(ranges):
            if idx == range_set:
                mapping[class_name] = new_idx
        # assert class_name in mapping
    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping

def get_subclass_label_mapping(ranges):
    def label_mapping(classes, class_to_idx):
        return subclass_label_mapping(classes, class_to_idx, ranges=ranges)
    return label_mapping

def setup_run(args):

    if args.local_rank == 0:
        run = wandb.init(
            config=args, name=args.save_dir.replace("results/", ''), save_code=True,
        )
    else:
        run = None

    return run

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # paddle.set_cuda_rng_state(seed)

class Logger:
    """Write console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def setup_logger(output=None):
    if output is None:
        return

    if output.endswith('.txt') or output.endswith('.log'):
        fpath = output
    else:
        fpath = osp.join(output, 'log.txt')

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime('-%Y-%m-%d-%H-%M-%S')

    sys.stdout = Logger(fpath)

def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.shape[0]

    # How many classes in target and whats the number of them
    unq, unq_cnt = np.unique(target.cpu(), return_counts=True)
    total_class = {int(unq[i]): unq_cnt[i] for i in range(len(unq))}     # dict to record class and corresponding num {class: class_num}

    output = F.softmax(output, axis=1)

    class_acc = {int(unq[i]): 0 for i in range(len(unq))}  # {class: 0}
    _, pred = output.topk(maxk, axis=1, largest=True, sorted=True)    # output values=[batch_size,maxk]  indices
    pred = pred.t()
    correct = pred.equal(target.reshape([1, -1]).expand_as(pred))

    for label, prediction in zip(target, pred.t()):
        if label == prediction[:1]:
            class_acc[int(label)] += 1


    res = []
    for k in topk: # (1,5)
        correct_k = paddle.cast(correct[:k].contiguous().reshape([-1]), dtype='float').sum(0)
        res.append(correct_k*100.0 / batch_size)

    if len(res) == 1:
        return res[0], class_acc   # res[0] 保存top1的acc，以此类推，topk=(1,5)则res[1]中保存tok[1]即top5的acc
    else:
        return (res[0], res[1], correct[0], pred[0], class_acc)



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # batch_average_acc e.g. 0.45
        self.sum += val * n  # usually batch_size in DL
        self.count += n
        self.avg = self.sum / self.count

