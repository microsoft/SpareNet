# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse


def get_args_from_command_line():
    """
    config the parameter
    """
    parser = argparse.ArgumentParser(description="The argument parser of R2Net runner")

    # choose model
    parser.add_argument("--model", type=str, default="sparenet", help="sparenet, atlasnet, msn, grnet")

    # choose train mode
    parser.add_argument("--gan", dest="gan", help="use gan", action="store_true", default=False)

    # choose load model
    parser.add_argument("--weights", dest="weights", help="Initialize network from the weights file", default=None)

    # setup gpu
    parser.add_argument("--gpu", dest="gpu_id", help="GPU device to use", default="0", type=str)

    # setup workdir
    parser.add_argument("--workdir", dest="workdir", help="where to save files", default=None)
    return parser.parse_args()


def main():
    args = get_args_from_command_line()

    # Set GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # update config
    from configs.base_config import cfg, cfg_from_file, cfg_update

    if args.gan:
        cfg_from_file("configs/" + args.model + "_gan.yaml")
    else:
        cfg_from_file("configs/" + args.model + ".yaml")
    output_dir = cfg_update(args)

    # Set up folders for logs and checkpoints
    if not os.path.exists(cfg.DIR.logs):
        os.makedirs(cfg.DIR.logs)
    from utils.misc import set_logger

    logger = set_logger(os.path.join(cfg.DIR.logs, "log.txt"))
    logger.info("save into dir: %s" % cfg.DIR.logs)

    # Start train/inference process
    if args.gan:
        runners = __import__("runners." + args.model + "_gan_runner")
        module = getattr(runners, args.model + "_gan_runner")
        model = getattr(module, args.model + "GANRunner")(cfg, logger)

    else:
        runners = __import__("runners." + args.model + "_runner")
        module = getattr(runners, args.model + "_runner")
        model = getattr(module, args.model + "Runner")(cfg, logger)

    model.runner()


if __name__ == "__main__":
    main()
