import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict
import parser

import torch
from torch.utils import data as torch_data
from torch import distributed as dist

from torchdrug import core, utils
from torchdrug.utils import comm

logger = logging.getLogger(__file__)


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], cfg.dataset["class"], cfg.task.model["class"],
                               time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    context_args = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in context_args:
        parser.add_argument("--%s" % var, required=True)
    context_args, unparsed = parser.parse_known_args()
    context_args = {k: utils.literal_eval(v) for k, v in context_args._get_kwargs()}

    with open(args.config, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context_args)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)

    # !Update command line input
    # Please note that there shouldn't be any duplicate keys in each subkey

    def overwrite_cfg_by_cmd_line(d, unparsed_str):
        # DFS the nested dict recursively
        for k, v in d.items():
            if isinstance(v, dict):
                overwrite_cfg_by_cmd_line(v, unparsed_str)
            elif f'--{k} ' in unparsed_str or f'--{k}=' in unparsed_str:
                # Update value by cmd
                parser.add_argument(f"--{k}", type=type(v))
                _arg, unparsed = parser.parse_known_args()
                _arg = {k: utils.literal_eval(v) for k, v in _arg._get_kwargs()}
                d[k] = _arg[k]
                unparsed_str = ' '.join(unparsed)

    overwrite_cfg_by_cmd_line(cfg, ' '.join(unparsed))
    return args, cfg


def build_solver(cfg, dataset):
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    if "fast_test" in cfg:
        if comm.get_rank() == 0:
            logger.warning("Quick test mode on. Only evaluate on %d samples for valid / test." % cfg.fast_test)
        g = torch.Generator()
        g.manual_seed(1024)
        valid_set = torch_data.random_split(valid_set, [cfg.fast_test, len(valid_set) - cfg.fast_test], generator=g)[0]
        test_set = torch_data.random_split(test_set, [cfg.fast_test, len(test_set) - cfg.fast_test], generator=g)[0]
    if hasattr(dataset, "num_relation"):
        cfg.task.model.num_relation = dataset.num_relation

    task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint)

    return solver
