# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Sequence, Union
import numpy as np
import sklearn.metrics
from numpy import ndarray
import scipy.stats

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging.meters import safe_round
import scipy

l1_loss = torch.nn.L1Loss(reduction='mean')
l2_loss = torch.nn.MSELoss(reduction='mean')
kl_loss = torch.nn.KLDivLoss(reduction='mean')


def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> ndarray:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))


def mean_absolute_error(target: Sequence[float],
                        prediction: Sequence[float]) -> ndarray:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> Union[ndarray, float]:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total


def get_acc(correct, total, round=4):
    if total is None or total == 0:
        return 0.
    try:
        return safe_round(correct / total, round)
    except OverflowError:
        return float('inf')


def _cross_entropy_pytorch(logits, target, ignore_index=None, reduction='mean'):
    lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    correct = torch.sum(lprobs.argmax(1) == target)
    return F.nll_loss(lprobs, target, ignore_index=ignore_index, reduction=reduction), correct

import time

@register_criterion('rna_twod_mlm')
class RNATwoDMaskedLMCriterion(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu
        # self.ltime = 0

    def forward(self, model, sample, reduce=True):
        # print("type:",type(sample))
        # print("sample like:",sample)
        # if self.ltime == 0:
        #     print("waiting for next sample")
        #     self.ltime = time.time()
        # else:
        #     print("one sample time:",time.time()-self.ltime)
        #     self.ltime = time.time()

        # print("input shape like:",sample['net_input']['src_tokens'].shape)
        # print("input shape like:",sample['net_input']['twod_tokens'].shape)
        logging_output = {}

        loss = torch.Tensor([0]).to(sample['target'].device)

        extra_only = False
        masked_only = True
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        targets = sample['target']
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        logits, ni, _ = model(**sample['net_input'], is_twod=True, masked_tokens=masked_tokens, 
                          masked_only=masked_only, extra_only=extra_only)
        del ni,_

        mlm_loss, mlm_correct = _cross_entropy_pytorch(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='mean',
            ignore_index=self.padding_idx,
        )
        mlm_loss = mlm_loss * sample['nsentences']

        loss = mlm_loss

        logging_output.update(
            {
                'mlm_loss': mlm_loss.data,
                'mlm_correct': utils.item(mlm_correct.data),
                'mlm_sample_size': sample_size,
                'loss': loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['nsentences'],
            }
        )

        del logits

        returned_sample_size = sample['nsentences']
        return loss, returned_sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """
        Aggregate logging outputs from data parallel training.
        """

        nsentences = float(sum(log['nsentences'] for log in logging_outputs))

        """loss/*_loss is the logged loss value over (num_)updates, corresponding to args.log_interval
        sample_size here is the number of total masked tokens
        ppl is the 2**loss, here the loss is the 'mean' loss for each tokens
        
        meter is the center log component for stroring values
        metrics's log_scalar/log_derived contributes to the meter's logged value
        """

        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)  # 'loss' must be logged as required
        metrics.log_scalar('loss', loss_sum / nsentences / math.log(2), weight=nsentences,
                            round=5)  # stored in meter

        mlm_loss_sum = sum(log.get('mlm_loss', 0) for log in logging_outputs)
        metrics.log_scalar('mlm_loss', mlm_loss_sum / nsentences / math.log(2), weight=nsentences, round=5)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['mlm_loss'].avg))
        mlm_correct_sum = float(sum(log.get("mlm_correct", 0) for log in logging_outputs))
        mlm_sample_size_sum = sum(
            log.get('mlm_sample_size', 0) for log in logging_outputs)  # the total maksed tokens
        metrics.log_scalar('mlm_correct', mlm_correct_sum / nsentences, weight=nsentences, priority=100)
        metrics.log_scalar('mlm_sample_size', mlm_sample_size_sum / nsentences, weight=nsentences, round=3,
            priority=100)
        metrics.log_derived('acc_mlm',
            lambda meters: get_acc(meters['mlm_correct'].avg, meters['mlm_sample_size'].avg, round=3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        # Criterions can override this, but for convenience we also try
        # to automatically map argparse.Namespace keys to corresponding
        # arguments in the __init__.
        init_args = {}
        import inspect
        for p in inspect.signature(cls).parameters.values():
            if (
                    p.kind == p.POSITIONAL_ONLY
                    or p.kind == p.VAR_POSITIONAL
                    or p.kind == p.VAR_KEYWORD
            ):
                # we haven't implemented inference for these argument types,
                # but PRs welcome :)
                raise NotImplementedError('{} not supported'.format(p.kind))

            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}

            if p.name == 'task':
                init_args['task'] = task
            elif hasattr(args, p.name):
                init_args[p.name] = getattr(args, p.name)
            elif p.default != p.empty:
                pass  # we'll use the default value
            else:
                raise NotImplementedError(
                    'Unable to infer Criterion arguments, please implement '
                    '{}.build_criterion'.format(cls.__name__)
                )
        return cls(**init_args)
