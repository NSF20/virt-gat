# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Doc String
"""
import sys
import os
import logging
import numpy as np
import math

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L

from pgl.utils import paddle_helper
from pgl.layers.conv import gat

def linear(X, hidden_size, name, with_bias=True):
    """linear"""
    fan_in=X.shape[-1]
    bias_bound = 1.0 / math.sqrt(fan_in)
    if with_bias:
        b_init = F.initializer.UniformInitializer(low=-bias_bound, high=bias_bound)
        fc_bias_attr = F.ParamAttr(initializer=b_init, name="%s_b" % name)
    else:
        fc_bias_attr = False
        
    negative_slope = math.sqrt(5)
    gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
    std = gain / math.sqrt(fan_in)
    weight_bound = math.sqrt(3.0) * std
    w_init = F.initializer.UniformInitializer(low=-weight_bound, high=weight_bound)
    fc_w_attr = F.ParamAttr(initializer=w_init, name="%s_w" % name)

    output = L.fc(X,
        hidden_size,
        param_attr=fc_w_attr,
        name=name,
        bias_attr=fc_bias_attr)
    return output

def layer_norm(feature, name=""):
    lay_norm_attr=F.ParamAttr(
            name="attr_%s" % name,
            initializer=F.initializer.ConstantInitializer(value=1))
    lay_norm_bias=F.ParamAttr(
            name="bias_%s" % name,
            initializer=F.initializer.ConstantInitializer(value=0))

    feature = L.layer_norm(feature, 
                           param_attr=lay_norm_attr,
                           bias_attr=lay_norm_bias)

    return feature

def copy_send(src_feat, dst_feat, edge_feat):
    """doc"""
    return src_feat["h"]


def mean_recv(feat):
    """doc"""
    return L.sequence_pool(feat, pool_type="average")


def sum_recv(feat):
    """doc"""
    return L.sequence_pool(feat, pool_type="sum")


def max_recv(feat):
    """doc"""
    return L.sequence_pool(feat, pool_type="max")


def lstm_recv(feat):
    """doc"""
    hidden_dim = 128
    forward, _ = L.dynamic_lstm(
        input=feat, size=hidden_dim * 4, use_peepholes=False)
    output = L.sequence_last_step(forward)
    return output

def gin(gw, feature, hidden_size, act, name):
    """doc"""
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, sum_recv)
    self_feature = feature
    output = self_feature + neigh_feature

    output = linear(output, hidden_size, name)
    # Residual
    output = output + feature
    return output

def graphsage_mean(gw, feature, hidden_size, act, name):
    """doc"""

    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, mean_recv)
    self_feature = feature
    output = L.concat([self_feature, neigh_feature], axis=1)

    output = linear(output, hidden_size, name)
    
    output = L.l2_normalize(output, axis=1)
    return output

def graphsage_bow(gw, feature, hidden_size, act, name):
    """doc"""
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, mean_recv)
    self_feature = feature
     
    output = self_feature + neigh_feature#], axis=1)
    output = L.l2_normalize(output, axis=1)
    return output

def transformer_conv(gw,
        feature,
        hidden_size,
        activation,
        name,
        num_heads=1,
        attn_drop=0,
        edge_feature=None,
        concat=True,
        is_test=False):
    '''transformer_gat_pgl
    '''
    def send_attention(src_feat, dst_feat, edge_feat):
        if edge_feat is None or not edge_feat:
            output = src_feat["k_h"] * dst_feat["q_h"]
            output = F.layers.reduce_sum(output, -1)
            output = output / (hidden_size ** 0.5)
            return {"alpha": output, "v": src_feat["v_h"]}   # batch x h     batch x h x feat
        else:
            edge_feat = edge_feat["edge"]
            edge_feat = F.layers.reshape(edge_feat, [-1, num_heads, hidden_size])
            output = (src_feat["k_h"] + edge_feat) * dst_feat["q_h"]
            output = F.layers.reduce_sum(output, -1)
            output = output / (hidden_size ** 0.5)
            return {"alpha": output, "v": (src_feat["v_h"] + edge_feat)}   # batch x h     batch x h x feat

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["v"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h
        
        if attn_drop > 1e-15:
            alpha = F.layers.dropout(
                alpha,
                dropout_prob=attn_drop,
                is_test=is_test,
                dropout_implementation="upscale_in_train")
        h = h * alpha
        h = F.layers.lod_reset(h, old_h)
        h = F.layers.sequence_pool(h, "sum")
        if concat:
            h = F.layers.reshape(h, [-1, num_heads * hidden_size])
        else:
            h = F.layers.reduce_mean(h, dim=1)
        return h
    
#     stdv = math.sqrt(6.0 / (feature.shape[-1] + hidden_size * num_heads))
#     q_w_attr=F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-stdv, high=stdv))
    q_w_attr=F.ParamAttr(initializer=F.initializer.XavierInitializer(),
            name=name + "_q_w")
    q_bias_attr=F.ParamAttr(initializer=F.initializer.ConstantInitializer(0.0),
            name=name + "q_bias")
    q = F.layers.fc(feature,
                    hidden_size * num_heads,
                    name=name + '_q_weight',
                    param_attr=q_w_attr,
                    bias_attr=q_bias_attr)
#     k_w_attr=F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-stdv, high=stdv))
    k_w_attr=F.ParamAttr(initializer=F.initializer.XavierInitializer(),
            name=name + "k_w")
    k_bias_attr=F.ParamAttr(initializer=F.initializer.ConstantInitializer(0.0),
            name=name + "k_bias")
    k = F.layers.fc(feature,
                         hidden_size * num_heads,
                       name=name + '_k_weight',
                       param_attr=k_w_attr,
                       bias_attr=k_bias_attr)
#     v_w_attr=F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-stdv, high=stdv))
    v_w_attr=F.ParamAttr(initializer=F.initializer.XavierInitializer(),
            name=name + "v_w")
    v_bias_attr=F.ParamAttr(initializer=F.initializer.ConstantInitializer(0.0),
            name=name + "v_bias")
    v = F.layers.fc(feature,
                         hidden_size * num_heads,
                       name=name + '_v_weight',
                       param_attr=v_w_attr,
                       bias_attr=v_bias_attr)
    
    reshape_q = F.layers.reshape(q, [-1, num_heads, hidden_size])
    reshape_k = F.layers.reshape(k, [-1, num_heads, hidden_size])
    reshape_v = F.layers.reshape(v, [-1, num_heads, hidden_size])

    msg = gw.send(
        send_attention,
        nfeat_list=[("q_h", reshape_q), ("k_h", reshape_k),
                    ("v_h", reshape_v)],
        efeat_list=edge_feature)
    output = gw.recv(msg, reduce_attention)

    return output


def simple_gat(gw, feature, hidden_size, activation, name, num_heads=1):
    def send_attention(src_feat, dst_feat, edge_feat):
        output = src_feat['left_a'] + dst_feat['right_a']
        output = L.leaky_relu(output, alpha=0.2)
        #  output = L.tanh(output)
        return {'alpha': output, 'h': src_feat['h']}

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        alpha = paddle_helper.sequence_softmax(alpha)
        h = msg["h"]
        old_h = h
        h = L.reshape(h, [-1, num_heads, hidden_size])
        alpha = L.reshape(alpha, [-1, num_heads, 1])

        h = h * alpha
        h = L.reshape(h, [-1, num_heads * hidden_size])
        h = L.lod_reset(h, old_h)
        return L.sequence_pool(h, "sum")

    ft = L.fc(feature,
             hidden_size * num_heads,
             bias_attr=False,
             param_attr=F.ParamAttr(name=name + '_weight'))

    left_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_l_A')
    right_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_r_A')

    reshape_ft = L.reshape(ft, [-1, num_heads, hidden_size])
    left_a_value = L.reduce_sum(reshape_ft * left_a, -1)
    right_a_value = L.reduce_sum(reshape_ft * right_a, -1)

    msg = gw.send(send_attention,
                nfeat_list=[("h", ft), 
                            ("left_a", left_a_value), 
                            ("right_a", right_a_value)
                        ]
                )

    #  alpha = msg['alpha']
    #  alpha = L.lod_reset(alpha, gw._edge_uniq_dst_count)
    #  alpha = paddle_helper.sequence_softmax(alpha)

    output = gw.recv(msg, reduce_attention)

    #  return output, alpha
    return output


