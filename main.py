import argparse
import logging
import os
import socket
import sys
import random
import numpy as np
# import paddle
import paddle
# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from utils.logger import (
    logging_config
)

from configs import get_cfg

from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
# from RSCFed.RSCFedManager import RSCFedManager
# from FedIRM.FedIRMManager import FedIRMManager

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str,
                        help="specify add which type of config")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # paddle.set_cuda_rng_state(seed)

if __name__ == "__main__":
    # initialize distributed computing (MPI)
    # parse python script input parameters

    #----------loading personalized params-----------------#
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args.config_file)
    #### set up cfg ####
    # default cfg
    cfg = get_cfg()

    # add registered cfg
    # some arguments that are needed by build_config come from args.
    cfg.setup(args)

    # Build config once again
    #cfg.setup(args)
    cfg.mode = 'standalone'

    cfg.server_index = -1
    cfg.client_index = -1
    seed = cfg.seed
    process_id = 0
    # show ultimate config
    logging.info(dict(cfg))

    #-------------------customize the process name-------------------
    str_process_name = cfg.algorithm + " (standalone):" + str(process_id)
    #setproctitle.setproctitle(str_process_name)

    logging_config(args=cfg, process_id=process_id)

    # loggcing.info("In Fed Con model construction, model is {}, {}, {}".format(
    #     cfg.model, type(cfg.model), cfg.model == 'simple-cnn'
    # ))

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if cfg.wandb_offline:
        os.environ['WANDB_MODE'] = 'dryrun'

    # Set the random seed. The np.random seed determines the dataset partition.

    # We fix these two, so that we can reproduce the result.
    set_random_seed(seed)
   
    paddle.seed(seed)
    # paddle.set_cuda_rng_state(seed)
    paddle.set_flags({"FLAGS_cudnn_deterministic":True})

    device = paddle.set_device("gpu:" + str(cfg.gpu_index) if paddle.is_compiled_with_cuda() else "cpu")
    if cfg.record_tool == 'wandb':
        import wandb
        wandb.init(config=cfg, name=cfg.wandb_name,
                   project=cfg.wandb_project)
    if cfg.algorithm == 'FedAvg':
        fedavg_manager = FedAVGManager(device, cfg)
        fedavg_manager.train()
    else:
        raise NotImplementedError
    wandb.finish()
    