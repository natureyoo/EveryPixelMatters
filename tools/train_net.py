# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader, make_data_loader_source, make_data_loader_target
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train_base
from fcos_core.modeling.detector import build_detection_model
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank, is_pytorch_1_1_0_or_later, all_gather
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
from torch.utils.tensorboard import SummaryWriter


def train(cfg, local_rank, distributed):
    writer = SummaryWriter('runs/{}'.format(cfg.OUTPUT_DIR))
    ##########################################################################
    ############################# Initial Model ##############################
    ##########################################################################
    model = {}
    device = torch.device(cfg.MODEL.DEVICE)

    backbone = build_backbone(cfg).to(device)
    fcos = build_rpn(cfg, backbone.out_channels).to(device)

    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
        fcos = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fcos)

    ##########################################################################
    #################### Initial Optimizer and Scheduler #####################
    ##########################################################################
    optimizer = {}
    optimizer["backbone"] = make_optimizer(cfg, backbone, name='backbone')
    optimizer["fcos"] = make_optimizer(cfg, fcos, name='fcos')
    scheduler = {}
    scheduler["backbone"] = make_lr_scheduler(cfg, optimizer["backbone"], name='backbone')
    scheduler["fcos"] = make_lr_scheduler(cfg, optimizer["fcos"], name='fcos')

    if distributed:
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )
        fcos = torch.nn.parallel.DistributedDataParallel(
            fcos, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )

    ########################### Save Model to Dict ###########################
    ##########################################################################
    model["backbone"] = backbone
    model["fcos"] = fcos

    ##########################################################################
    ################################ Training ################################
    ##########################################################################
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader_source(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train_base(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        cfg,
        run_test,
        distributed,
        writer
    )

    return model


def run_test(cfg, model, distributed):
    model_test = {}
    if distributed:
        model_test["backbone"] = model["backbone"].module
        model_test["fcos"] = model["fcos"].module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_name = cfg.DATASETS.TEST[0]
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    results = inference(
        model_test,
        data_loaders_val[0],
        dataset_name=dataset_name,
        iou_types=iou_types,
        box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        output_folder=output_folder,
    )
    synchronize()
    results = all_gather(results)
    # import pdb; pdb.set_trace()
    return results


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    main()
