#-*- coding: utf-8 -*-
import os
import sys
sys.path.append("../")
import json
import copy
import numpy as np
import tqdm
from collections import OrderedDict, namedtuple, defaultdict

import pgl
from pgl.utils.data.dataset import Dataset, StreamDataset, HadoopDataset

def load_vocab(vocab_file):
    name2id = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            name, idx = line.strip().split('\t')
            name2id[name] = int(idx)

    return name2id

class BaseParser(object):
    def __init__(self, config, **kwargs):
        self.config = config

    def __call__(self, line):
        raise NotImplementedError("not implemented!")

class NewParser(BaseParser):
    def __init__(self, config, **kwargs):
        self.config = config

    def __call__(self, line):
        if self.config.graph_pool_type == "virtual":
            return self.virt_parse(line)
        else:
            return self.base_parse(line)

    def base_parse(self, line):
        url, json_str = line.strip().split('\t')
        json_data = json.loads(json_str)

        gdata = {}

        num_nodes = len(json_data[self.config.slots[0]])
        # node features
        slot_feats = {}
        slot_info = {}
        for slot in self.config.slots:
            feat = np.array(json_data[slot], dtype="int64").reshape(-1, 1)
            feat[np.where(feat < 0)] = 0
            slot_feats[slot] = feat
            slot_info[slot] = np.arange(1, num_nodes + 1, dtype="int64")

        # graph features
        g_slot_feats = {}
        for slot in self.config.graph_slots:
            g_slot_feats[slot] = np.array([json_data[slot]], dtype="int64").reshape(-1, 1)

        gdata.update(g_slot_feats)

        # edges
        edges = copy.deepcopy(json_data['edges'])
        if self.config.symmetry:
            r_edges = np.array(json_data["edges"])[:, [1, 0]].tolist()
            edges.extend(r_edges)
        if self.config.self_loop:
            for i in range(num_nodes):
                edges.append([i, i])

        edges = np.array(edges, dtype="int64").reshape(-1,2)

        max_id = np.max(np.array(json_data['edges'], dtype="int64").reshape(-1,))
        if max_id >= num_nodes:
            raise ValueError("the max edge ID (%s) is larger than num_nodes (%s)." % (max_id, num_nodes))
        gdata['edges'] = edges
        gdata['label'] = json_data['label']
        gdata['num_edges'] = len(edges)
        gdata['num_nodes'] = num_nodes
        gdata['slot_feats'] = slot_feats
        gdata['slot_info'] = slot_info
        gdata['url'] = url

        return gdata

    def virt_parse(self, line):
        url, json_str = line.strip().split('\t')
        json_data = json.loads(json_str)

        gdata = {}

        num_nodes = len(json_data[self.config.slots[0]])
        # node features
        slot_feats = {}
        slot_info = {}
        for slot in self.config.slots:
            feat = np.array(json_data[slot] + [0], dtype="int64").reshape(-1, 1)
            feat[np.where(feat < 0)] = 0
            slot_feats[slot] = feat
            slot_info[slot] = np.arange(1, num_nodes + 2, dtype="int64")

        # graph features
        g_slot_feats = {}
        for slot in self.config.graph_slots:
            g_slot_feats[slot] = np.array([json_data[slot]], dtype="int64").reshape(-1, 1)

        gdata.update(g_slot_feats)

        virt_edges = []
        virt_node_id = num_nodes
        for nid in range(num_nodes):
            virt_edges.append([nid, virt_node_id])

        num_nodes = num_nodes + 1  # for virtual node

        edges = virt_edges + json_data['edges']
        # edges
        #  edges = copy.deepcopy(json_data['edges'])
        if self.config.symmetry:
            r_edges = np.array(edges)[:, [1, 0]].tolist()
            edges.extend(r_edges)
        if self.config.self_loop:
            for i in range(num_nodes):
                edges.append([i, i])

        edges = np.array(edges, dtype="int64").reshape(-1,2)

        max_id = np.max(np.array(json_data['edges'], dtype="int64").reshape(-1,))
        if max_id >= num_nodes:
            raise ValueError("the max edge ID (%s) is larger than num_nodes (%s)." % (max_id, num_nodes))
        gdata['edges'] = edges
        gdata['label'] = json_data['label']
        gdata['num_edges'] = len(edges)
        gdata['num_nodes'] = num_nodes
        gdata['slot_feats'] = slot_feats
        gdata['slot_info'] = slot_info
        gdata['url'] = url

        return gdata

def str2graph(line, config, vocab=None):
    url, json_str = line.strip().split('\t')
    json_data = json.loads(json_str)

    graph_data = {}

    num_nodes = len(json_data['node'])
    edges = copy.deepcopy(json_data['edges'])
    if config.self_loop:
        for i in range(num_nodes):
            edges.append([i, i])

    edges = np.array(edges, dtype="int64").reshape(-1,2)

    max_id = np.max(np.array(json_data['edges'], dtype="int64").reshape(-1,))
    if max_id >= num_nodes:
        raise ValueError("the max edge ID (%s) is larger than num_nodes (%s)." % (max_id, num_nodes))

    #  node_feat = np.random.rand(num_nodes, 64)
    #  node_feat = parse_nfeat(json_data['node'], vocab)
    #  node_feat = np.array(node_feat, dtype="int64").reshape(-1, 1)

    slot_feats, slot_info = parse_nfeat2(json_data['node'], config.slots, config.slots_size)

    #  node_feat = parse_nfeat3(json_data['node'], config.slots)
    #  node_feat = np.array(node_feat, dtype="float32")
    if config.graph_slots is not None:
        graph_data.update(
                parse_gfeat(json_data, config.graph_slots, config.graph_slots_size))

    graph_data['label'] = json_data['label']
    graph_data['edges'] = edges
    graph_data['num_edges'] = len(edges)
    graph_data['num_nodes'] = num_nodes
    #  graph_data['node_feat'] = node_feat
    graph_data['slot_feats'] = slot_feats
    graph_data['slot_info'] = slot_info
    graph_data['url'] = url

    return graph_data

def parse_nfeat(node_infos, tag_vocab):
    nfeat = []
    for n_info in node_infos:
        tag_name = n_info['feature']['name']
        nfeat.append([tag_vocab[tag_name]])
    return nfeat

def parse_gfeat(graph_infos, slots, slots_size):
    slot_feats = defaultdict(list)

    for slot, max_size in zip(slots, slots_size):
        try:
            feat_value = int(graph_infos[slot])
        except Exception as e:
            feat_value = 0

        if feat_value >= max_size or feat_value < 0:
            feat_value = 0

        slot_feats[slot].append(feat_value)

    for slot in slots:
        slot_feats[slot] = np.array(slot_feats[slot], dtype="int64").reshape(-1, 1)

    return slot_feats


def parse_nfeat2(node_infos, slots, slots_size):
    slot_feats = defaultdict(list)
    slot_info = defaultdict(list)
    for slot, max_size in zip(slots, slots_size):
        for n_info in node_infos:
            try:
                feat_value = n_info['feature'][slot]
            except Exception as e:
                #  log.info(e)
                #  log.info("slot %s has no feat value, set to 0" % slot)
                feat_value = 0

            if type(feat_value) is not int:
                feat_value = 0

            if feat_value >= max_size or feat_value < 0:
                feat_value = 0

            slot_feats[slot].append(feat_value)
            length = len(slot_info[slot])
            slot_info[slot].append(length + 1)

    for slot in slots:
        slot_feats[slot] = np.array(slot_feats[slot], dtype="int64").reshape(-1,1)
        slot_info[slot] = np.array(slot_info[slot], dtype="int64").reshape(-1, )

    return slot_feats, slot_info

def parse_nfeat3(node_infos, slots):
    """
    regard the node feature as input feature directly
    """

    nfeat = []
    for n_info in node_infos:
        feat = []
        for slot in slots:
            try:
                feat_value = n_info['feature'][slot]
            except Exception as e:
                feat_value = -1
            feat_value += 1  # index 0 is for missing feature
            feat.append(feat_value)
        nfeat.append(feat)

    return nfeat

def test_label(config):
    parser = NewParser(config)
    y_0 = 0
    y_1 = 0
    for line in tqdm.tqdm(sys.stdin):
        gdata = parser(line)
        #  gdata = str2graph(line, config)
        if int(gdata['label']) == 0:
            y_0 += 1
        elif int(gdata["label"]) == 1:
            y_1 += 1
        else:
            print("label is ", gdata['label'], type(gdata['label']))
    print("y_0: %s | y_1: %s" % (y_0, y_1))

def test_example(config):
    parser = NewParser(config)
    with open('old_format_data', 'r') as f:
        for line in f:
            gdata = str2graph(line, config, None)

    with open('new_format_data', 'r') as f:
        for line in f:
            new_gdata = parser(line)

    assert gdata['num_edges'] == new_gdata['num_edges']
    assert gdata['num_nodes'] == new_gdata['num_nodes']
    assert gdata['label'] == new_gdata['label']

    if (gdata['edges'] != new_gdata['edges']).any():
        print("edges")

    for key, value in gdata['slot_feats'].items():
        if (gdata['slot_feats'][key] != value).any():
            print(key)
 

if __name__=="__main__":
    config = prepare_config("./config.yaml")
    test_label(config)
