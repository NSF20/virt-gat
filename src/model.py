#-*- coding: utf-8 -*-
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
from pgl.graph_wrapper import BatchGraphWrapper
from propeller import log
import propeller.paddle as propeller
import paddle.fluid as F
import paddle.fluid.layers as L

from utils.config import prepare_config, make_dir
from dataset import Inputs
import layers as GNNlayers


def copy_send(src_feat, dst_feat, edge_feat):
    return src_feat["h"]

def mean_recv(feat):
    return L.sequence_pool(feat, pool_type="average")

class GNNModel(propeller.train.Model):
    def __init__(self, hparam, mode, run_config):
        self.hparam = hparam
        self.mode = mode
        self.is_test = True if self.mode != propeller.RunMode.TRAIN else False
        self.run_config = run_config

    def node_id_embedding(self, node_index):
        """get node id embedding"""
        id_emb = L.embedding(
            input=node_index,
            size=(self.hparam.tag_size, self.hparam.emb_size),
            param_attr=F.ParamAttr(name="embedding"))

        return id_emb

    def get_slot_embed(self, slot, length, slot_index, lod):
        slot_embed = L.embedding(
                input=slot_index,
                size=(length, self.hparam.emb_size),
                param_attr=F.ParamAttr(name="embed_%s" % slot))

        lod = L.cast(lod, dtype="int32")
        lod = L.reshape(L.reshape(lod, [1, -1]), [-1])
        lod.stop_gradient = True
        #  L.Print(slot_embed, message="%s" % slot, summarize=20)
        #  L.Print(lod, message="%s_lod: " % slot, summarize=20)
        slot_embed = L.lod_reset(slot_embed, lod)
        return slot_embed

    def preprocess_nfeat(self, input_dict):
        # node id embedding feature
        #  feature = [self.node_id_embedding(input_dict['node_index'])]

        #---- slot embedding feature ---#
        feature = []
        for slot, length in zip(self.run_config.slots, self.run_config.slots_size):
            slot_embed = self.get_slot_embed(slot,
                                      length,
                                      input_dict[slot],
                                      input_dict["%s_info" % slot])

            slot_embed = L.sequence_pool(slot_embed, pool_type="sum")
            slot_embed = L.softsign(slot_embed)
            feature.append(slot_embed)

        feature = L.sum(feature)

        if self.hparam.softsign:
            feature = L.softsign(feature)

        #---- fc embedding feature ----#
        #  feature = input_dict['nfeat']

        return feature

    def process_graph_embedding(self, input_dict):
        feature = []
        for slot, length in zip(self.run_config.graph_slots, self.run_config.graph_slots_size):
            slot_embed = self.get_slot_embed(slot,
                                              length,
                                              input_dict[slot],
                                              input_dict['%s_ginfo' % slot])
            slot_embed = L.sequence_pool(slot_embed, pool_type="sum")
            slot_embed = L.softsign(slot_embed)
            feature.append(slot_embed)

        feature = L.sum(feature)

        if self.hparam.softsign:
            feature = L.softsign(feature)

        return feature

    def forward(self, input_dict):
        gw = BatchGraphWrapper(input_dict['num_nodes'],
                               input_dict['num_edges'],
                               input_dict['edges'])

        feature = self.preprocess_nfeat(input_dict)

        for layer in range(self.hparam.num_layers):
            if layer == self.hparam.num_layers - 1:
                act = None
            else:
                act = 'leaky_relu'

            feature = getattr(GNNlayers, self.hparam.layer_type)(
                    gw,
                    feature,
                    self.hparam.hidden_size,
                    act,
                    name="%s_%s" % (self.hparam.layer_type, layer))

            if self.hparam.l2_norm:
                feature = L.l2_normalize(feature, axis=1)

        if self.hparam.graph_pool_type == "virtual":
            pooled_h = self.virt_node_pool(gw, feature)
        else:
            pooled_h = pgl.layers.graph_pooling(gw, feature, self.hparam.graph_pool_type)

        if self.hparam.graph_slots is not None:
            gfeature = self.process_graph_embedding(input_dict)

            if self.hparam.gfeat_mode == "concat":
                pooled_h = L.concat([pooled_h, gfeature])
            else:
                pooled_h = L.sum([pooled_h, gfeature])

        pooled_h = L.dropout(
            pooled_h,
            self.hparam.dropout_prob,
            dropout_implementation="upscale_in_train")

        logits = GNNlayers.linear(pooled_h, self.hparam.num_class, "final_fc")

        probs = L.softmax(logits)

        return [probs, input_dict['url'], input_dict['v_labels']]

    def virt_node_pool(self, graph_wrapper, node_features, pool_type=None):
        if pool_type == 'gat':
            log.info("virt pool type is gat")
            node_features = pgl.layers.gat(graph_wrapper,
                                           node_features,
                                           hidden_size=self.args.hidden_size//8,
                                           activation="relu",
                                           name="gin_gat",
                                           num_heads=8,
                                           feat_drop=0.0,
                                           attn_drop=0.0)
        else:
            msg = graph_wrapper.send(copy_send, nfeat_list=[('h', node_features)])
            node_features = graph_wrapper.recv(msg, mean_recv)

        graph_feat = L.lod_reset(node_features, graph_wrapper.graph_lod)
        graph_feat = L.sequence_last_step(graph_feat)
        return graph_feat

    def loss(self, predictions, label):
        probs = predictions[0]

        loss = L.cross_entropy(input=probs, label=label)
        loss = L.reduce_mean(loss)

        return loss

    def backward(self, loss):
        optimizer = F.optimizer.Adam(learning_rate=self.hparam.lr)
        optimizer.minimize(loss)

    def metrics(self, predictions, label):
        result = {}
        probs = predictions[0]
        acc = L.accuracy(input=probs, label=label)
        acc = propeller.metrics.Mean(acc)
        result['acc'] = acc


        y_pred = L.reshape(L.argmax(probs, axis=1), [-1, 1])
        result['hard_mse'] = propeller.metrics.MSE(label, y_pred)

        y_pred = L.one_hot(y_pred, depth=self.hparam.num_class)
        y_true = L.one_hot(label, depth=self.hparam.num_class)

        for i in range(self.hparam.num_class):
            result['P_%s' % i] = propeller.metrics.Precision(
                    y_true[:, i], y_pred[:, i])
            result['R_%s' % i] = propeller.metrics.Recall(
                    y_true[:, i], y_pred[:, i])
            result['F1_%s' % i] = propeller.metrics.F1(
                    y_true[:, i], y_pred[:, i])

        return result

class RegModel(propeller.train.Model):
    def __init__(self, hparam, mode, run_config):
        self.hparam = hparam
        self.mode = mode
        self.is_test = True if self.mode != propeller.RunMode.TRAIN else False
        self.run_config = run_config
        # 0.5, 1.5, 2.5
        self.threshold_list = [i + 0.5 for i in range(self.hparam.num_class - 1)]
        if self.hparam.label_norm:
            self.threshold_list = [ i/3 for i in self.threshold_list]
        log.info("threshold_list: %s" % self.threshold_list)

    def node_id_embedding(self, node_index):
        """get node id embedding"""
        id_emb = L.embedding(
            input=node_index,
            size=(self.hparam.tag_size, self.hparam.emb_size),
            param_attr=F.ParamAttr(name="embedding"))

        return id_emb

    def get_slot_embed(self, slot, length, slot_index, lod):
        slot_embed = L.embedding(
                input=slot_index,
                size=(length, self.hparam.emb_size),
                param_attr=F.ParamAttr(name="embed_%s" % slot))

        lod = L.cast(lod, dtype="int32")
        lod = L.reshape(L.reshape(lod, [1, -1]), [-1])
        lod.stop_gradient = True
        slot_embed = L.lod_reset(slot_embed, lod)
        return slot_embed

    def preprocess_nfeat(self, input_dict):
        # node id embedding feature
        #  feature = self.node_id_embedding(node_feat)

        # slot embedding feature
        feature = []
        for slot, length in zip(self.run_config.slots, self.run_config.slots_size):
            slot_embed = self.get_slot_embed(slot,
                                      length,
                                      input_dict[slot],
                                      input_dict["%s_info" % slot])

            slot_embed = L.sequence_pool(slot_embed, pool_type="sum")
            slot_embed = L.softsign(slot_embed)
            feature.append(slot_embed)

        feature = L.sum(feature)
        if self.hparam.softsign:
            feature = L.softsign(feature)
        
        # fc embedding feature
        #  feature = input_dict['nfeat']

        return feature

    def forward(self, input_dict):
        #  num_nodes, num_edges, edges, node_feat, v_labels, urls = features
        gw = BatchGraphWrapper(input_dict['num_nodes'],
                               input_dict['num_edges'],
                               input_dict['edges'])

        feature = self.preprocess_nfeat(input_dict)

        for layer in range(self.hparam.num_layers):
            if layer == self.hparam.num_layers - 1:
                act = None
            else:
                act = 'leaky_relu'

            feature = getattr(GNNlayers, self.hparam.layer_type)(
                    gw,
                    feature,
                    self.hparam.hidden_size,
                    act,
                    name="%s_%s" % (self.hparam.layer_type, layer))

            if self.hparam.l2_norm:
                feature = L.l2_normalize(feature, axis=1)

        pooled_h = pgl.layers.graph_pooling(gw, feature, self.hparam.graph_pool_type)

        pooled_h = L.dropout(
            pooled_h,
            self.hparam.dropout_prob,
            is_test=self.is_test,
            dropout_implementation="upscale_in_train")

        logits = GNNlayers.linear(pooled_h, 1, "final_fc")

        if self.hparam.label_norm:
            logits = L.sigmoid(logits)
            #  logits = L.softsign(logits)

        if self.mode == propeller.RunMode.PREDICT:
            if self.hparam.num_class == 2:
                y_pred = self.score2binary(logits)
            else:
                y_pred = self.score2class(logits)
            y_pred = L.argmax(y_pred, axis=1)
            return [logits, input_dict['url'], input_dict['v_labels'], y_pred]
        else:
            return [logits, input_dict['url'], input_dict['v_labels']]

    def loss(self, predictions, label):
        probs = predictions[0]

        label = L.cast(label, dtype="float32")
        if self.hparam.label_norm:
            label = label / (self.hparam.num_class - 1)

        loss = L.square_error_cost(input=probs, label=label)

        loss = L.reduce_mean(loss)

        return loss

    def backward(self, loss):
        optimizer = F.optimizer.Adam(learning_rate=self.hparam.lr)
        optimizer.minimize(loss)

    def metrics(self, predictions, label):
        result = {}
        probs = predictions[0]
        f_label = L.cast(label, dtype="float32")
        result['mse'] = propeller.metrics.MSE(f_label, probs)
        result['mcrmse'] = propeller.metrics.MCRMSE(f_label, probs)

        res = self.calc_metrics(probs, label)
        result.update(res)

        return result

    def calc_metrics(self, probs, label):
        y_true = L.one_hot(label, depth=self.hparam.num_class)
        if self.hparam.num_class == 2:
            y_pred = self.score2binary(probs)
        else:
            y_pred = self.score2class(probs)

        result = {}
        #  pred = L.reshape(L.argmax(L.concat(y_pred, axis=1), axis=1), [-1, 1])
        pred = L.reshape(L.argmax(y_pred, axis=1), [-1, 1])
        result['hard_mse'] = propeller.metrics.MSE(label, pred)

        for i in range(self.hparam.num_class):
            result['P_%s' % i] = propeller.metrics.Precision(
                    y_true[:, i], y_pred[:, i])

            result['R_%s' % i] = propeller.metrics.Recall(
                    y_true[:, i], y_pred[:, i])

            result['F1_%s' % i] = propeller.metrics.F1(
                    y_true[:, i], y_pred[:, i])

        return result

    def score2class(self, probs):
        b_pred = []
        y_pred = []

        for threshold in self.threshold_list:
            b_pred.append(probs <= threshold)

        # label0 <= 0.25
        y_pred0 = L.cast(b_pred[0], dtype="int64")
        y_pred.append(y_pred0)

        # 0.25 < label1 <= 0.5
        y_pred1 = L.cast(L.logical_and(L.logical_not(b_pred[0]), b_pred[1]), dtype="int64")
        y_pred.append(y_pred1)

        # 0.5 < label1 <= 0.75
        y_pred2 = L.cast(L.logical_and(L.logical_not(b_pred[1]), b_pred[2]), dtype="int64")
        y_pred.append(y_pred2)

        # label3 > 0.75
        y_pred3 = L.cast(L.logical_not(b_pred[2]), dtype="int64")
        y_pred.append(y_pred3)

        y_pred = L.concat(y_pred, axis=1)
        return y_pred

    def score2binary(self, probs):
        b_pred = []
        y_pred = []

        for threshold in self.threshold_list:
            b_pred.append(probs <= threshold)

        y_pred0 = L.cast(b_pred[0], dtype="int64")
        y_pred.append(y_pred0)

        y_pred1 = L.cast(L.logical_not(b_pred[0]), dtype="int64")
        y_pred.append(y_pred1)

        y_pred = L.concat(y_pred, axis=1)
        return y_pred


