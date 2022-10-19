import paddle
import sys
sys.path.append("../")
import os
import argparse
import traceback
import re
import io
import json
import yaml
import time 
import logging
from tqdm import tqdm
import numpy as np
from collections import namedtuple

import pgl
from pgl.utils import paddle_helper
from pgl.utils.data.dataloader import Dataloader
from propeller import log
log.setLevel(logging.DEBUG)
import propeller.paddle as propeller
from propeller.paddle.data import Dataset as PDataset

import paddle.fluid as F
import paddle.fluid.layers as L

from utils.config import prepare_config, make_dir
from utils.logger import prepare_logger, log_to_file
from utils.util import int82strarr
import dataset as DS
from dataset import CollateFn
import model as M
import exporter

def multi_epoch_dataloader(loader, epochs):
    def _worker():
        for i in range(epochs):
            log.info("BEGIN: epoch %s ..." % i)
            for batch in loader():
                yield batch
            log.info("END: epoch %s ..." % i)
    return _worker

def train(args):
    log.info("loading data")
    train_ds = getattr(DS, args.dataset_type)(args, "train", args.train_data_path)
    collate_fn = getattr(DS, args.collate_fn)(args)
    train_loader = Dataloader(train_ds, 
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         shuffle=args.shuffle,
                         stream_shuffle_size=args.shuffle_size,
                         collate_fn=collate_fn)

    train_loader = multi_epoch_dataloader(train_loader, args.epochs)
    train_loader = PDataset.from_generator_func(train_loader)

    valid_ds = getattr(DS, args.dataset_type)(args, "dev", args.dev_data_path)
    dev_loader = Dataloader(valid_ds, 
                         batch_size=args.batch_size,
                         num_workers=1,
                         collate_fn=collate_fn)
    dev_loader = PDataset.from_generator_func(dev_loader)

    if args.warm_start_from:
        # warm start setting
        def _fn(v):
            if not isinstance(v, F.framework.Parameter):
                return False
            if os.path.exists(os.path.join(args.warm_start_from, v.name)):
                return True
            else:
                return False
        ws = propeller.WarmStartSetting(
                predicate_fn=_fn,
                from_dir=args.warm_start_from)
    else:
        ws = None

    def cmp_fn(old, new):
        if old['eval'][args.metrics] - new['eval'][args.metrics] > 0:
            return True
        else:
            return False

    exp = exporter.BestResultExporter(args.output_dir, cmp_fn)

    propeller.train.train_and_eval(
            model_class_or_model_fn=getattr(M, args.model_type),
            params=args,
            run_config=args,
            train_dataset=train_loader,
            eval_dataset={"eval": dev_loader},
            warm_start_setting=ws,
            exporters=[exp],
            )

    log.info("final eval best result: %.6f" \
            % exp._best_result['eval'][args.metrics])

def evaluate(args):
    dev_ds = getattr(DS, args.dataset_type)(args, "dev", args.test_data_path)
    collate_fn = getattr(DS, args.collate_fn)(args, "dev")
    dev_loader = Dataloader(dev_ds, 
                         batch_size=args.batch_size,
                         num_workers=1,
                         collate_fn=collate_fn)
    dev_loader = PDataset.from_generator_func(dev_loader)

    ws = propeller.WarmStartSetting(
            predicate_fn=lambda v: os.path.exists(os.path.join(args.model_path_for_infer, v.name)),
            from_dir=args.model_path_for_infer)
    est = propeller.Learner(getattr(M, args.model_type), args, args, ws)

    result = est.evaluate(dev_loader)

def reg_infer(args):
    test_ds = getattr(DS, args.dataset_type)(args, "test", args.test_data_path)
    collate_fn = getattr(DS, args.collate_fn)(args, "test")
    test_loader = Dataloader(test_ds, 
                         batch_size=args.batch_size,
                         num_workers=1,
                         collate_fn=collate_fn)
    test_loader = PDataset.from_generator_func(test_loader)

    est = propeller.Learner(getattr(M, args.model_type), args, args)

    output_path = args.model_path_for_infer.replace("checkpoints/", "outputs/")
    make_dir(output_path)
    if args.prefix is not None:
        filename = os.path.join(output_path, "%s_predict.txt" % args.prefix)
    else:
        filename = os.path.join(output_path, "predict.txt")
    log.info("saving result to %s" % filename)

    cc = 0
    with open(filename, 'w', buffering=1) as f:
        f.write("y_true\tscore\ty_pred\turl\n")
        for probs, int8_urls, v_labels, y_pred in est.predict(test_loader,
                ckpt_path=args.model_path_for_infer, split_batch=False):

            int8_urls = np.array(int8_urls, dtype="int8")
            str_urls = int82strarr(int8_urls)
            for url, p, ground, pred in zip(str_urls, probs, v_labels, y_pred):
                str_p = "%.4f" % p[0]
                f.write("%s\t%s\t%s\t%s\n" % (ground, str_p, pred, url))

                cc += 1
                if cc % 10000 == 0:
                    log.info("%s examples have been predicted." % cc)

def cls_infer(args):
    test_ds = getattr(DS, args.dataset_type)(args, "test", args.test_data_path)
    collate_fn = getattr(DS, args.collate_fn)(args, "test")
    test_loader = Dataloader(test_ds, 
                         batch_size=args.batch_size,
                         num_workers=1,
                         collate_fn=collate_fn)
    test_loader = PDataset.from_generator_func(test_loader)

    est = propeller.Learner(getattr(M, args.model_type), args, args)

    output_path = args.model_path_for_infer.replace("checkpoints/", "outputs/")
    make_dir(output_path)
    if args.prefix is not None:
        filename = os.path.join(output_path, "%s_predict.txt" % args.prefix)
    else:
        filename = os.path.join(output_path, "predict.txt")
    log.info("saving result to %s" % filename)

    cc = 0
    with open(filename, 'w', buffering=1) as f:
        heads = ["p%s" % i for i in range(args.num_class)]
        heads = '\t'.join(heads)
        f.write("y_true\t%s\ty_pred\turl\n" % heads)
        for probs, int8_urls, v_labels in est.predict(test_loader, 
                ckpt_path=args.model_path_for_infer, split_batch=False):

            int8_urls = np.array(int8_urls, dtype="int8")
            str_urls = int82strarr(int8_urls)

            for url, p, ground in zip(str_urls, probs, v_labels):
                str_p = []
                for i in p:
                    str_p.append("%.4f" % i)
                str_p = '\t'.join(str_p)
                pred = np.argmax(p)
                f.write("%s\t%s\t%s\t%s\n" % (ground, str_p, pred, url))

                cc += 1
                if cc % 2000 == 0:
                    log.info("%s examples have been predicted." % cc)

def hadoop_infer(args):
    test_ds = DS.StdinDataset(args, "test", args.test_data_path)
    collate_fn = getattr(DS, args.collate_fn)(args, "test")
    test_loader = Dataloader(test_ds, 
                         batch_size=args.batch_size,
                         num_workers=1,
                         collate_fn=collate_fn)
    test_loader = PDataset.from_generator_func(test_loader)

    est = propeller.Learner(getattr(M, args.model_type), args, args)

    cc = 0
    #  heads = ["p%s" % i for i in range(args.num_class)]
    #  heads = '\t'.join(heads)
    #  f.write("y_true\t%s\ty_pred\turl\n" % heads)
    for probs, int8_urls, v_labels in est.predict(test_loader, 
            ckpt_path=args.model_path_for_infer, split_batch=False):

        int8_urls = np.array(int8_urls, dtype="int8")
        str_urls = int82strarr(int8_urls)

        for url, p, ground in zip(str_urls, probs, v_labels):
            str_p = []
            for i in p:
                str_p.append("%.4f" % i)
            str_p = '\t'.join(str_p)
            pred = np.argmax(p)
            sys.stdout.write("%s\t%s\t%s\t%s\n" % (ground, str_p, pred, url))

        cc += 1
        if cc % 10000 == 0:
            log.info("%s examples have been predicted." % cc)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--task_name", type=str, default="task_name")
    args = parser.parse_args()

    if args.mode == "cls_infer":
        config = prepare_config(args.config, isCreate=False, isSave=False)
        cls_infer(config)
    elif args.mode == "reg_infer":
        config = prepare_config(args.config, isCreate=False, isSave=False)
        reg_infer(config)
    elif args.mode == "hadoop_infer":
        config = prepare_config(args.config, isCreate=False, isSave=False)
        hadoop_infer(config)
    else:
        config = prepare_config(args.config, isCreate=True, isSave=True)
        log_to_file(log, config.log_dir)
        train(config)
