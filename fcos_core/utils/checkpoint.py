# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from fcos_core.utils.model_serialization import load_state_dict
from fcos_core.utils.c2_model_loading import load_c2_format
from fcos_core.utils.imports import import_file
from fcos_core.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded and "model_backbone" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model_backbone"] = self.model["backbone"].state_dict()
        data["model_fcos"] = self.model["fcos"].state_dict()

        if self.cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if self.cfg.MODEL.ADV.USE_DIS_P7:
                data["model_dis_P7"] = self.model["dis_P7"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P6:
                data["model_dis_P6"] = self.model["dis_P6"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P5:
                data["model_dis_P5"] = self.model["dis_P5"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P4:
                data["model_dis_P4"] = self.model["dis_P4"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P3:
                data["model_dis_P3"] = self.model["dis_P3"].state_dict()

        if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if self.cfg.MODEL.ADV.USE_DIS_P7:
                data["model_dis_P7_CA"] = self.model["dis_P7_CA"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P6:
                data["model_dis_P6_CA"] = self.model["dis_P6_CA"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P5:
                data["model_dis_P5_CA"] = self.model["dis_P5_CA"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P4:
                data["model_dis_P4_CA"] = self.model["dis_P4_CA"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P3:
                data["model_dis_P3_CA"] = self.model["dis_P3_CA"].state_dict()

        if self.cfg.MODEL.ADV.USE_DIS_CONDITIONAL:
            if self.cfg.MODEL.ADV.USE_DIS_P7:
                data["model_dis_P7_Cond"] = self.model["dis_P7_Cond"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P6:
                data["model_dis_P6_Cond"] = self.model["dis_P6_Cond"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P5:
                data["model_dis_P5_Cond"] = self.model["dis_P5_Cond"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P4:
                data["model_dis_P4_Cond"] = self.model["dis_P4_Cond"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P3:
                data["model_dis_P3_Cond"] = self.model["dis_P3_Cond"].state_dict()

        if self.cfg.MODEL.ADV.USE_DIS_HEAD:
            if self.cfg.MODEL.ADV.USE_DIS_P7:
                data["model_dis_P7_HA"] = self.model["dis_P7_HA"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P6:
                data["model_dis_P6_HA"] = self.model["dis_P6_HA"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P5:
                data["model_dis_P5_HA"] = self.model["dis_P5_HA"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P4:
                data["model_dis_P4_HA"] = self.model["dis_P4_HA"].state_dict()
            if self.cfg.MODEL.ADV.USE_DIS_P3:
                data["model_dis_P3_HA"] = self.model["dis_P3_HA"].state_dict()

        if self.optimizer is not None:
            if self.cfg.MODEL.ADV.USE_DIS_GLOBAL:
                if self.cfg.MODEL.ADV.USE_DIS_P7:
                    data["optimizer_dis_P7"] = self.optimizer["dis_P7"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P6:
                    data["optimizer_dis_P6"] = self.optimizer["dis_P6"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P5:
                    data["optimizer_dis_P5"] = self.optimizer["dis_P5"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P4:
                    data["optimizer_dis_P4"] = self.optimizer["dis_P4"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P3:
                    data["optimizer_dis_P3"] = self.optimizer["dis_P3"].state_dict()

            if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
                if self.cfg.MODEL.ADV.USE_DIS_P7:
                    data["optimizer_dis_P7_CA"] = self.optimizer["dis_P7_CA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P6:
                    data["optimizer_dis_P6_CA"] = self.optimizer["dis_P6_CA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P5:
                    data["optimizer_dis_P5_CA"] = self.optimizer["dis_P5_CA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P4:
                    data["optimizer_dis_P4_CA"] = self.optimizer["dis_P4_CA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P3:
                    data["optimizer_dis_P3_CA"] = self.optimizer["dis_P3_CA"].state_dict()

            if self.cfg.MODEL.ADV.USE_DIS_CONDITIONAL:
                if self.cfg.MODEL.ADV.USE_DIS_P7:
                    data["optimizer_dis_P7_Cond"] = self.optimizer["dis_P7_Cond"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P6:
                    data["optimizer_dis_P6_Cond"] = self.optimizer["dis_P6_Cond"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P5:
                    data["optimizer_dis_P5_Cond"] = self.optimizer["dis_P5_Cond"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P4:
                    data["optimizer_dis_P4_Cond"] = self.optimizer["dis_P4_Cond"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P3:
                    data["optimizer_dis_P3_Cond"] = self.optimizer["dis_P3_Cond"].state_dict()

            if self.cfg.MODEL.ADV.USE_DIS_HEAD:
                if self.cfg.MODEL.ADV.USE_DIS_P7:
                    data["optimizer_dis_P7_HA"] = self.optimizer["dis_P7_HA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P6:
                    data["optimizer_dis_P6_HA"] = self.optimizer["dis_P6_HA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P5:
                    data["optimizer_dis_P5_HA"] = self.optimizer["dis_P5_HA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P4:
                    data["optimizer_dis_P4_HA"] = self.optimizer["dis_P4_HA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P3:
                    data["optimizer_dis_P3_HA"] = self.optimizer["dis_P3_HA"].state_dict()

        if self.scheduler is not None:
            if self.cfg.MODEL.ADV.USE_DIS_GLOBAL:
                if self.cfg.MODEL.ADV.USE_DIS_P7:
                    data["scheduler_dis_P7"] = self.scheduler["dis_P7"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P6:
                    data["scheduler_dis_P6"] = self.scheduler["dis_P6"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P5:
                    data["scheduler_dis_P5"] = self.scheduler["dis_P5"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P4:
                    data["scheduler_dis_P4"] = self.scheduler["dis_P4"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P3:
                    data["scheduler_dis_P3"] = self.scheduler["dis_P3"].state_dict()

            if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
                if self.cfg.MODEL.ADV.USE_DIS_P7:
                    data["scheduler_dis_P7_CA"] = self.scheduler["dis_P7_CA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P6:
                    data["scheduler_dis_P6_CA"] = self.scheduler["dis_P6_CA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P5:
                    data["scheduler_dis_P5_CA"] = self.scheduler["dis_P5_CA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P4:
                    data["scheduler_dis_P4_CA"] = self.scheduler["dis_P4_CA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P3:
                    data["scheduler_dis_P3_CA"] = self.scheduler["dis_P3_CA"].state_dict()

            if self.cfg.MODEL.ADV.USE_DIS_CONDITIONAL:
                if self.cfg.MODEL.ADV.USE_DIS_P7:
                    data["scheduler_dis_P7_Cond"] = self.scheduler["dis_P7_Cond"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P6:
                    data["scheduler_dis_P6_Cond"] = self.scheduler["dis_P6_Cond"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P5:
                    data["scheduler_dis_P5_Cond"] = self.scheduler["dis_P5_Cond"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P4:
                    data["scheduler_dis_P4_Cond"] = self.scheduler["dis_P4_Cond"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P3:
                    data["scheduler_dis_P3_Cond"] = self.scheduler["dis_P3_Cond"].state_dict()

            if self.cfg.MODEL.ADV.USE_DIS_HEAD:
                if self.cfg.MODEL.ADV.USE_DIS_P7:
                    data["scheduler_dis_P7_HA"] = self.scheduler["dis_P7_HA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P6:
                    data["scheduler_dis_P6_HA"] = self.scheduler["dis_P6_HA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P5:
                    data["scheduler_dis_P5_HA"] = self.scheduler["dis_P5_HA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P4:
                    data["scheduler_dis_P4_HA"] = self.scheduler["dis_P4_HA"].state_dict()
                if self.cfg.MODEL.ADV.USE_DIS_P3:
                    data["scheduler_dis_P3_HA"] = self.scheduler["dis_P3_HA"].state_dict()

        # data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, load_dis=True, load_opt_sch=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, load_dis)

        if load_opt_sch:
            if "optimizer_fcos" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))

                self.optimizer["backbone"].load_state_dict(checkpoint.pop("optimizer_backbone"))
                self.optimizer["fcos"].load_state_dict(checkpoint.pop("optimizer_fcos"))

                if self.cfg.MODEL.ADV.USE_DIS_GLOBAL:
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        self.optimizer["dis_P7"].load_state_dict(checkpoint.pop("optimizer_dis_P7"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        self.optimizer["dis_P6"].load_state_dict(checkpoint.pop("optimizer_dis_P6"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        self.optimizer["dis_P5"].load_state_dict(checkpoint.pop("optimizer_dis_P5"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        self.optimizer["dis_P4"].load_state_dict(checkpoint.pop("optimizer_dis_P4"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        self.optimizer["dis_P3"].load_state_dict(checkpoint.pop("optimizer_dis_P3"))

                if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        self.optimizer["dis_P7_CA"].load_state_dict(checkpoint.pop("optimizer_dis_P7_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        self.optimizer["dis_P6_CA"].load_state_dict(checkpoint.pop("optimizer_dis_P6_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        self.optimizer["dis_P5_CA"].load_state_dict(checkpoint.pop("optimizer_dis_P5_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        self.optimizer["dis_P4_CA"].load_state_dict(checkpoint.pop("optimizer_dis_P4_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        self.optimizer["dis_P3_CA"].load_state_dict(checkpoint.pop("optimizer_dis_P3_CA"))

                if self.cfg.MODEL.ADV.USE_DIS_CONDITIONAL:
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        self.optimizer["dis_P7_Cond"].load_state_dict(checkpoint.pop("optimizer_dis_P7_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        self.optimizer["dis_P6_Cond"].load_state_dict(checkpoint.pop("optimizer_dis_P6_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        self.optimizer["dis_P5_Cond"].load_state_dict(checkpoint.pop("optimizer_dis_P5_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        self.optimizer["dis_P4_Cond"].load_state_dict(checkpoint.pop("optimizer_dis_P4_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        self.optimizer["dis_P3_Cond"].load_state_dict(checkpoint.pop("optimizer_dis_P3_Cond"))

                if self.cfg.MODEL.ADV.USE_DIS_HEAD:
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        self.optimizer["dis_P7_HA"].load_state_dict(checkpoint.pop("optimizer_dis_P7_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        self.optimizer["dis_P6_HA"].load_state_dict(checkpoint.pop("optimizer_dis_P6_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        self.optimizer["dis_P5_HA"].load_state_dict(checkpoint.pop("optimizer_dis_P5_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        self.optimizer["dis_P4_HA"].load_state_dict(checkpoint.pop("optimizer_dis_P4_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        self.optimizer["dis_P3_HA"].load_state_dict(checkpoint.pop("optimizer_dis_P3_HA"))
            else:
                self.logger.info(
                    "No optimizer found in the checkpoint. Initializing model from scratch"
                )

            if "scheduler_fcos" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))

                self.scheduler["backbone"].load_state_dict(checkpoint.pop("scheduler_backbone"))
                self.scheduler["fcos"].load_state_dict(checkpoint.pop("scheduler_fcos"))

                if self.cfg.MODEL.ADV.USE_DIS_GLOBAL:
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        self.scheduler["dis_P7"].load_state_dict(checkpoint.pop("scheduler_dis_P7"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        self.scheduler["dis_P6"].load_state_dict(checkpoint.pop("scheduler_dis_P6"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        self.scheduler["dis_P5"].load_state_dict(checkpoint.pop("scheduler_dis_P5"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        self.scheduler["dis_P4"].load_state_dict(checkpoint.pop("scheduler_dis_P4"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        self.scheduler["dis_P3"].load_state_dict(checkpoint.pop("scheduler_dis_P3"))

                if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        self.scheduler["dis_P7_CA"].load_state_dict(checkpoint.pop("scheduler_dis_P7_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        self.scheduler["dis_P6_CA"].load_state_dict(checkpoint.pop("scheduler_dis_P6_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        self.scheduler["dis_P5_CA"].load_state_dict(checkpoint.pop("scheduler_dis_P5_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        self.scheduler["dis_P4_CA"].load_state_dict(checkpoint.pop("scheduler_dis_P4_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        self.scheduler["dis_P3_CA"].load_state_dict(checkpoint.pop("scheduler_dis_P3_CA"))

                if self.cfg.MODEL.ADV.USE_DIS_CONDITIONAL:
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        self.scheduler["dis_P7_Cond"].load_state_dict(checkpoint.pop("scheduler_dis_P7_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        self.scheduler["dis_P6_Cond"].load_state_dict(checkpoint.pop("scheduler_dis_P6_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        self.scheduler["dis_P5_Cond"].load_state_dict(checkpoint.pop("scheduler_dis_P5_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        self.scheduler["dis_P4_Cond"].load_state_dict(checkpoint.pop("scheduler_dis_P4_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        self.scheduler["dis_P3_Cond"].load_state_dict(checkpoint.pop("scheduler_dis_P3_Cond"))

                if self.cfg.MODEL.ADV.USE_DIS_HEAD:
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        self.scheduler["dis_P7_HA"].load_state_dict(checkpoint.pop("scheduler_dis_P7_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        self.scheduler["dis_P6_HA"].load_state_dict(checkpoint.pop("scheduler_dis_P6_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        self.scheduler["dis_P5_HA"].load_state_dict(checkpoint.pop("scheduler_dis_P5_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        self.scheduler["dis_P4_HA"].load_state_dict(checkpoint.pop("scheduler_dis_P4_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        self.scheduler["dis_P3_HA"].load_state_dict(checkpoint.pop("scheduler_dis_P3_HA"))
            else:
                self.logger.info(
                    "No scheduler found in the checkpoint. Initializing model from scratch"
                )

        # return any further checkpoint data
        return checkpoint

    def _load_model(self, checkpoint, load_dis=True):
        if "model_backbone" in checkpoint:
            # load checkpoint of our model
            load_state_dict(self.model["backbone"], checkpoint.pop("model_backbone"))
            load_state_dict(self.model["fcos"], checkpoint.pop("model_fcos"))
            if self.cfg.MODEL.ADV.USE_DIS_GLOBAL and load_dis:
                if "model_dis_P3" in checkpoint or "model_dis_P4" in checkpoint or "model_dis_P5" in checkpoint or "model_dis_P6" in checkpoint or "model_dis_P7" in checkpoint:
                    self.logger.info("Global alignment discriminator checkpoint found. Initializing model from the checkpoint")
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        load_state_dict(self.model["dis_P7"], checkpoint.pop("model_dis_P7"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        load_state_dict(self.model["dis_P6"], checkpoint.pop("model_dis_P6"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        load_state_dict(self.model["dis_P5"], checkpoint.pop("model_dis_P5"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        load_state_dict(self.model["dis_P4"], checkpoint.pop("model_dis_P4"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        load_state_dict(self.model["dis_P3"], checkpoint.pop("model_dis_P3"))
                else:
                    self.logger.info(
                        "No global discriminator found in the checkpoint. Initializing model from scratch"
                    )

            if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE and load_dis:
                if "model_dis_P3_CA" in checkpoint or "model_dis_P4_CA" in checkpoint or "model_dis_P5_CA" in checkpoint or "model_dis_P6_CA" in checkpoint or "model_dis_P7_CA" in checkpoint:
                    self.logger.info("Center-aware alignment discriminator checkpoint found. Initializing model from the checkpoint")
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        load_state_dict(self.model["dis_P7_CA"], checkpoint.pop("model_dis_P7_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        load_state_dict(self.model["dis_P6_CA"], checkpoint.pop("model_dis_P6_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        load_state_dict(self.model["dis_P5_CA"], checkpoint.pop("model_dis_P5_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        load_state_dict(self.model["dis_P4_CA"], checkpoint.pop("model_dis_P4_CA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        load_state_dict(self.model["dis_P3_CA"], checkpoint.pop("model_dis_P3_CA"))
                else:
                    self.logger.info(
                        "No center-aware discriminator found in the checkpoint. Initializing model from scratch"
                    )

            if self.cfg.MODEL.ADV.USE_DIS_CONDITIONAL and load_dis:
                if "model_dis_P3_Cond" in checkpoint or "model_dis_P4_Cond" in checkpoint or "model_dis_P5_Cond" in checkpoint or "model_dis_P6_Cond" in checkpoint or "model_dis_P7_Cond" in checkpoint:
                    self.logger.info("Conditional alignment discriminator checkpoint found. Initializing model from the checkpoint")
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        load_state_dict(self.model["dis_P7_Cond"], checkpoint.pop("model_dis_P7_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        load_state_dict(self.model["dis_P6_Cond"], checkpoint.pop("model_dis_P6_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        load_state_dict(self.model["dis_P5_Cond"], checkpoint.pop("model_dis_P5_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        load_state_dict(self.model["dis_P4_Cond"], checkpoint.pop("model_dis_P4_Cond"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        load_state_dict(self.model["dis_P3_Cond"], checkpoint.pop("model_dis_P3_Cond"))
                else:
                    self.logger.info(
                        "No conditional discriminator found in the checkpoint. Initializing model from scratch"
                    )

            if self.cfg.MODEL.ADV.USE_DIS_HEAD and load_dis:
                if "model_dis_P3_HA" in checkpoint or "model_dis_P4_HA" in checkpoint or "model_dis_P5_HA" in checkpoint or "model_dis_P6_HA" in checkpoint or "model_dis_P7_HA" in checkpoint:
                    self.logger.info("Head alignment discriminator checkpoint found. Initializing model from the checkpoint")
                    if self.cfg.MODEL.ADV.USE_DIS_P7:
                        load_state_dict(self.model["dis_P7_HA"], checkpoint.pop("model_dis_P7_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P6:
                        load_state_dict(self.model["dis_P6_HA"], checkpoint.pop("model_dis_P6_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P5:
                        load_state_dict(self.model["dis_P5_HA"], checkpoint.pop("model_dis_P5_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P4:
                        load_state_dict(self.model["dis_P4_HA"], checkpoint.pop("model_dis_P4_HA"))
                    if self.cfg.MODEL.ADV.USE_DIS_P3:
                        load_state_dict(self.model["dis_P3_HA"], checkpoint.pop("model_dis_P3_HA"))
                else:
                    self.logger.info(
                        "No head discriminator found in the checkpoint. Initializing model from scratch"
                    )
        else:
            # load others, e.g., Imagenet pretrained pkl
            load_state_dict(self.model["backbone"], checkpoint.pop("model"))
