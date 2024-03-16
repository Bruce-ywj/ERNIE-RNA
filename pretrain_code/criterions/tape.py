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


@register_criterion('prot_tape')
class ProtTapeEvalCriterion(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training,
    which is used for PSSM prediciton
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        extra_only = True
        masked_only = False
        src_tokens = sample['input_ids'].detach().cpu().numpy()
        masked_tokens = sample['input_mask'].detach().cpu().numpy()

        _, extra = model(src_tokens=sample['input_ids'], features_only=True,
                            extra_only=extra_only, masked_only=masked_only)
        sample_size = float(len(extra['pooled_output']))

        sequence_output = extra['inner_states'][-1].transpose(0, 1)

        if self.task.args.eval_task in ['secondary_structure']:
            tape_loss, correct, total, pred_logits = model.classification_heads[self.task.args.eval_task](
                sequence_output, **sample)
        elif self.task.args.eval_task in ['contact_prediction']:
            if 'x_attns' in extra and extra['x_attns'] is not None:  # pairwise mlm output can be directed used
                tape_loss, correct, total, pred_logits = model.classification_heads[self.task.args.eval_task](
                    extra['x_attns'], **sample)
            else:
                tape_loss, correct, total, pred_logits = model.classification_heads[self.task.args.eval_task](
                    sequence_output, **sample)
            # tape_loss, correct, total, pred_logits = model.classification_heads[self.task.args.eval_task](
            #         sequence_output, **sample)
        else:
            tape_loss, correct, total, pred_logits = model.classification_heads[self.task.args.eval_task](
                extra['pooled_output'], **sample)

        tape_loss = tape_loss * sample_size

        if correct == None:
            assert self.task.args.eval_task in ['stability', 'fluorescence']
            logging_output = {
                'tape_loss': tape_loss.data,
                'sample_size': sample_size,
                'prediction': pred_logits.data.cpu().numpy(),
                'target_': sample['targets'].data.cpu().numpy(),
                'ntokens': float(sum(sample['protein_length'])),  # used for log wpb/wps
                'nsentences': sample_size,  # used for log bsz
                # 'total': utils.item(total.data),
            }
        else:
            logging_output = {
                'tape_loss': tape_loss.data,
                'correct': float(correct),
                'sample_size': float(total),
                'ntokens': float(sum(sample['protein_length'])),  # used for log wpb/wps
                'nsentences': sample_size,  # used for log bsz
                # 'total': utils.item(total.data),
            }

        if self.task.args.eval_task == 'contact_prediction':
            aupr = pred_logits * sample_size
            logging_output.update(
                {
                    'cmap_aupr': aupr,
                }
            )

        returned_sample_size = sample_size
        return tape_loss, returned_sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """
        Aggregate logging outputs from data parallel training.
        """

        nsentences = float(sum(log['nsentences'] for log in logging_outputs))

        if 'tape_loss' in logging_outputs[0]:  # TAPE's: fluorescence, stability
            loss_sum = sum(log.get('tape_loss', 0) for log in logging_outputs)
            sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
            metrics.log_scalar('sample_size', sample_size / nsentences, weight=sample_size, round=2,
                                priority=100)
            metrics.log_scalar('loss', loss_sum / nsentences / math.log(2), weight=sample_size,
                                round=5)  # stored in meter

            if 'target_' in logging_outputs[0]:
                targets_array = []
                prediction_array = []
                for log in logging_outputs:
                    targets_array.extend(log['target_'])
                    prediction_array.extend(log['prediction'])
                ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
                metrics.log_scalar('spearmanr', spearmanr(prediction_array, targets_array), weight=ntokens, round=5,
                                    priority=10)

            if 'correct' in logging_outputs[0]:
                correct_sum = float(sum(log.get("correct", 0) for log in logging_outputs))
                metrics.log_scalar('correct', correct_sum / nsentences, weight=sample_size, round=2,
                                    priority=100)
                if 'cmap_aupr' in logging_outputs[0]:
                    metrics.log_derived(
                        'cmap_precision_L5',
                        lambda meters: get_acc(meters['correct'].avg, meters['sample_size'].avg, round=3))

                    auprc = sum(log.get("cmap_aupr", 0) for log in logging_outputs)
                    metrics.log_scalar('cmap_AUPRC', auprc / nsentences, weight=sample_size, round=5)
                else:
                    metrics.log_derived(
                        'acc', lambda meters: get_acc(meters['correct'].avg, meters['sample_size'].avg, round=3),
                        priority=10)
        # else:
        #     loss_sum = sum(log.get('loss', 0) for log in logging_outputs)  # 'loss' must be logged as required
        #     metrics.log_scalar('loss', loss_sum / nsentences / math.log(2), weight=nsentences,
        #                        round=5)  # stored in meter

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False

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
