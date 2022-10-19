#-*- coding: utf-8 -*-
import os
import sys
sys.path.append("../")
import json
import time
import numpy as np
import glob
import argparse
import pickle
from collections import OrderedDict, namedtuple, defaultdict

import pgl
from pgl.utils.data.dataloader import Dataloader
from pgl.utils.data.dataset import Dataset, StreamDataset, HadoopDataset
from propeller import log

from utils.config import prepare_config, make_dir
from utils.logger import prepare_logger, log_to_file
from utils.util import strarr2int8arr, int82strarr
from data_parser import str2graph, load_vocab
import data_parser as DParser

def weighted_sampling(examples, num_class=4):
    total_labels = list(range(num_class))

    label_array = np.array([e['label'] for e in examples], dtype="int64")
    for i in total_labels:
        log.info("total examples of label==%s: %s" \
                % (i, np.sum(label_array == i)))

    sampler_weight = np.zeros(len(examples))
    for label in total_labels:
        cond = label_array == label
        sampler_weight[cond] = 1.0 / len(total_labels) / np.sum(cond)

    indices = np.random.choice(len(examples), 
                               len(examples), 
                               replace=True, 
                               p=sampler_weight)
    res = [examples[i] for i in indices]

    label_array = np.array([e['label'] for e in res], dtype="int64")
    for i in total_labels:
        log.info("After weighted sampling, total examples of label==%s: %s" \
                % (i, np.sum(label_array == i)))
    return res


class DemoHadoopDataset(HadoopDataset):
    def __init__(self, config, mode, data_path):
        super(DemoHadoopDataset, self).__init__(config.hadoop_bin, 
                                                config.fs_name, 
                                                config.fs_ugi)

        self.config = config
        self.mode = mode
        self.data_path = data_path
        #  self.tag_vocab = load_vocab(self.config.tag_vocab_file)
        self.tag_vocab = None

    def __iter__(self):
        for line in self._line_data_generator():
            graph_data = str2graph(line, self.config, self.tag_vocab)
            yield graph_data

    def _line_data_generator(self):
        paths = self.hadoop_util.ls(self.data_path)
        paths = sorted(paths)
        filelist = []

        #TODO: If I have multi machines, how to split data?
        # Refer to m2v_pslib

        for idx, filename in enumerate(paths):
            if idx % self._worker_info.num_workers != self._worker_info.fid:
                continue
            filelist.append(filename)

        for filename in filelist:
            with self.hadoop_util.open(filename, encoding='utf-8') as f:
                for line in f:
                    yield line

class NewFormatDataset(HadoopDataset):
    def __init__(self, config, mode, data_path):
        super(NewFormatDataset, self).__init__(config.hadoop_bin, 
                                                config.fs_name, 
                                                config.fs_ugi)

        self.config = config
        self.mode = mode
        self.data_path = data_path
        self.parser = DParser.NewParser(config)

    def __iter__(self):
        for line in self._line_data_generator():
            graph_data = self.parser(line)
            yield graph_data

    def _line_data_generator(self):
        paths = self.hadoop_util.ls(self.data_path)
        num_files = len(paths)
        log.info(f'dataset mode is {self.mode}, total {num_files} data files')
        paths = sorted(paths)
        filelist = []

        #TODO: If I have multi machines, how to split data?
        # Refer to m2v_pslib

        for idx, filename in enumerate(paths):
            if idx % self._worker_info.num_workers != self._worker_info.fid:
                continue
            filelist.append(filename)

        for filename in filelist:
            with self.hadoop_util.open(filename, encoding='utf-8') as f:
                for line in f:
                    yield line



class StdinDataset(StreamDataset):
    def __init__(self, config, mode, data_path):
        self.config = config
        self.mode = mode
        self.data_path = data_path
        self.parser = DParser.NewParser(config)

    def __iter__(self):
        for line in self._line_data_generator():
            #  graph_data = str2graph(line, self.config, vocab=None)
            graph_data = self.parser(line)
            yield graph_data
            #  except Exception as e:
            #      log.info(e)
                #  continue

    def _line_data_generator(self):
        for line in sys.stdin:
            yield line

class OldDataset(Dataset):
    def __init__(self, config, mode, data_path):
        self.config = config
        self.mode = mode
        self.data_path = data_path
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(line)

    def __getitem__(self, idx):
        line = self.data[idx]
        graph_data = str2graph(line, self.config, None)
        return graph_data



class LocalDataset(StreamDataset):
    def __init__(self, config, mode, data_path):
        self.config = config
        self.mode = mode
        self.data_path = data_path
        self.parser = DParser.NewParser(config)

        #  self.data = self._load_data()

    def __iter__(self):
        for line in self._line_data_generator():
            graph_data = self.parser(line)
            yield graph_data

    def _line_data_generator(self):
        data_path = os.path.join(self.data_path, "*")
        filelist = glob.glob(data_path)
        for filename in filelist:
            with open(filename, "r") as reader:
                for line in reader:
                    yield line

class CollateFn(object):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

    def __call__(self, batch_data):
        feed_dict = dict()
        num_nodes = []
        num_edges = []
        edges = []
        labels = []
        #  node_index = []
        url_list = []
        slot_feats = defaultdict(list)
        slot_info = defaultdict(list)
        offset = 0
        for graph_data in batch_data:
            num_nodes.append(graph_data['num_nodes'])
            num_edges.append(graph_data['num_edges'])
            edges.append(graph_data['edges'])
            labels.append(graph_data['label'])
            for slot in self.config.slots:
                slot_feats[slot].append(graph_data['slot_feats'][slot])
                slot_info[slot].append(graph_data['slot_info'][slot] + offset)
            offset += graph_data['num_nodes']
            url_list.append(graph_data['url'])

        feed_dict['num_nodes'] = np.array(num_nodes, dtype="int64")
        feed_dict['num_edges'] = np.array(num_edges, dtype="int64")
        feed_dict['edges'] = np.concatenate(edges).astype("int64")
        #  feed_dict['nfeat'] = np.concatenate(nfeat).astype("int64")
        feed_dict['v_labels'] = np.array(labels, dtype="int64").reshape(-1,)

        for slot in self.config.slots:
            feed_dict[slot] = np.concatenate(slot_feats[slot], axis=0)
            tmp = [np.array([0], dtype="int64")]
            tmp.extend(slot_info[slot])
            feed_dict["%s_info" % slot] = np.concatenate(tmp)
            #  assert len(feed_dict[slot]) == (len(feed_dict["%s_info" % slot]) - 1)
            #  assert len(feed_dict[slot]) > 0

        if self.mode == "test":
            feed_dict['url'] = strarr2int8arr(url_list).astype("int64")
        else:
            feed_dict['url'] = np.array([1,2,3], dtype="int8")  # fake, for speed up
            feed_dict['labels'] = np.array(labels, dtype="int64").reshape(-1,1)

        return feed_dict

class NfeatCollateFn(object):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

    def __call__(self, batch_data):
        feed_dict = dict()
        num_nodes = []
        num_edges = []
        edges = []
        labels = []
        nfeat = []
        url_list = []
        offset = 0
        for graph_data in batch_data:
            num_nodes.append(graph_data['num_nodes'])
            num_edges.append(graph_data['num_edges'])
            edges.append(graph_data['edges'])
            labels.append(graph_data['label'])
            nfeat.append(graph_data['node_feat'])
            offset += graph_data['num_nodes']
            url_list.append(graph_data['url'])

        feed_dict['num_nodes'] = np.array(num_nodes, dtype="int64")
        feed_dict['num_edges'] = np.array(num_edges, dtype="int64")
        feed_dict['edges'] = np.concatenate(edges).astype("int64")
        feed_dict['nfeat'] = np.concatenate(nfeat).astype("float32")
        feed_dict['v_labels'] = np.array(labels, dtype="int64").reshape(-1,)

        if self.mode == "test":
            feed_dict['url'] = strarr2int8arr(url_list).astype("int64")
        else:
            feed_dict['url'] = np.array([1,2,3], dtype="int8")  # fake, for speed up
            feed_dict['labels'] = np.array(labels, dtype="int64").reshape(-1,1)

        return feed_dict

class GfeatCollateFn(object):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

    def __call__(self, batch_data):
        feed_dict = dict()
        num_nodes = []
        num_edges = []
        edges = []
        labels = []
        #  node_index = []
        url_list = []
        slot_feats = defaultdict(list)
        slot_info = defaultdict(list)
        slot_gfeats = defaultdict(list)
        slot_ginfo = defaultdict(list)
        for slot in self.config.graph_slots:
            slot_ginfo[slot] = [0]
        offset = 0
        for graph_data in batch_data:
            num_nodes.append(graph_data['num_nodes'])
            num_edges.append(graph_data['num_edges'])
            edges.append(graph_data['edges'])
            labels.append(graph_data['label'])
            for slot in self.config.slots:
                slot_feats[slot].append(graph_data['slot_feats'][slot])
                slot_info[slot].append(graph_data['slot_info'][slot] + offset)
            for slot in self.config.graph_slots:
                slot_gfeats[slot].extend(graph_data[slot])
                slot_ginfo[slot].append(len(slot_gfeats[slot]))
            offset += graph_data['num_nodes']
            url_list.append(graph_data['url'])

        feed_dict['num_nodes'] = np.array(num_nodes, dtype="int64")
        feed_dict['num_edges'] = np.array(num_edges, dtype="int64")
        feed_dict['edges'] = np.concatenate(edges).astype("int64")
        #  feed_dict['nfeat'] = np.concatenate(nfeat).astype("int64")
        feed_dict['v_labels'] = np.array(labels, dtype="int64").reshape(-1,)

        for slot in self.config.slots:
            feed_dict[slot] = np.concatenate(slot_feats[slot], axis=0)
            tmp = [np.array([0], dtype="int64")]
            tmp.extend(slot_info[slot])
            feed_dict["%s_info" % slot] = np.concatenate(tmp)
            #  assert len(feed_dict[slot]) == (len(feed_dict["%s_info" % slot]) - 1)
            #  assert len(feed_dict[slot]) > 0

        for slot in self.config.graph_slots:
            feed_dict[slot] = np.array(slot_gfeats[slot], dtype="int64").reshape(-1, 1)
            feed_dict["%s_ginfo" % slot]  = np.array(slot_ginfo[slot], dtype="int64")

        if self.mode == "test":
            feed_dict['url'] = strarr2int8arr(url_list).astype("int64")
        else:
            feed_dict['url'] = np.array([1,2,3], dtype="int8")  # fake, for speed up
            feed_dict['labels'] = np.array(labels, dtype="int64").reshape(-1,1)

        return feed_dict



Inputs = namedtuple("inputs", 
        ["num_nodes", "num_edges", "edges"])

def test_data_format(config):
    #  new_ds = StdinDataset(config, 'test', config.test_data_path)

    old_ds = OldDataset(config, 'test', config.test_data_path)

    for i in range(1):
        print(old_ds[i]['url'])
        print(old_ds[i]['edges'].tolist())

def test_new_data_format(config):
    #  new_ds = StdinDataset(config, 'test', config.test_data_path)
    new_ds = NewFormatDataset(config, 'test', config.test_data_path)

    new_ds = iter(new_ds)

    for ex in new_ds:
        print(ex['url'])
        print(ex['edges'].tolist())
        break

def base_test(config):
    #  valid_ds = LocalDataset(config, 'train', config.train_data_path)
    ds = DemoHadoopDataset(config, 'train', config.train_data_path)
    loader = Dataloader(ds, 
                         batch_size=1,
                         num_workers=1,
                         collate_fn=GfeatCollateFn(config))

    #  import ipdb;ipdb.set_trace()
    for idx, batch in enumerate(loader):
        #  for slot in config.slots:
        #      print(slot, len(batch[slot]))
        #      print("%s_info" % slot, len(batch["%s_info" % slot]))
        for slot in config.graph_slots:
            print(slot, batch[slot])
        time.sleep(5)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    config = prepare_config(args.config, isCreate=False, isSave=False)

    if args.mode == 'new':
        test_new_data_format(config)
    else:
        test_data_format(config)

