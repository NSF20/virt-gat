#-*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
import os
import itertools
import six
import inspect
import abc
import logging

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L

from propeller.util import map_structure
from propeller.paddle.train import Saver
from propeller.types import InferenceSpec
from propeller.train.model import Model
from propeller.paddle.train.trainer import _build_net
from propeller.paddle.train.trainer import _build_model_fn
from propeller.types import RunMode
from propeller.types import ProgramPair
from propeller import log
log.setLevel(logging.DEBUG)
import propeller.paddle as propeller

class BestResultExporter(propeller.exporter.Exporter):
    """export saved model accordingto `cmp_fn`"""

    def __init__(self, export_dir, cmp_fn):
        """doc"""
        self._export_dir = export_dir
        self._best = None
        if isinstance(cmp_fn, tuple):
            self.cmp_fn = cmp_fn[0]
            self.is_export = cmp_fn[1]
        else:
            self.cmp_fn = cmp_fn
            self.is_export = False
        self._best_result = None
        self._epoch_count = 0

    def export(self, exe, program, eval_model_spec, eval_result, state):
        """doc"""
        log.debug('New evaluate result: %s \nold: %s' %
                  (repr(eval_result), repr(self._best)))
        if self._best is None and state['best_model'] is not None:
            self._best = state['best_model']
            log.debug('restoring best state %s' % repr(self._best))
            self._best_result = eval_result
        if self._best is None or self.cmp_fn(old=self._best, new=eval_result):

            log.debug('[Best Exporter]: export to %s' % self._export_dir)
            eval_program = program.train_program
            # FIXME: all eval datasets has same name/types/shapes now!!! so every eval program are the smae

            saver = Saver(
                self._export_dir, exe, program=program, max_ckpt_to_keep=1)
            eval_result = map_structure(float, eval_result)
            self._best = eval_result
            state['best_model'] = eval_result
            self._best_result = eval_result
            saver.save(state)
        else:
            log.debug('[Best Exporter]: skip step %s' % state.gstep)

        self._epoch_count += 1
        if self._epoch_count % 10 == 0:
            log.info("EPOCH_%s: %s" % (self._epoch_count, repr(self._best_result)))

