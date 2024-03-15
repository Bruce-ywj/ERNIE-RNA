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

from ..pmlm_utils import (
    mask_batch_target, 
    calc_batch_pair_probs_from_mlm_logits, 
    calc_batch_pair_targets_from_mlm_targets,
    calc_batch_mlm_lprobs_from_pair_logits,
    prune_batch_mlm_target
)

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

def _pair_loss(logits, target, masked_tokens, vocab_size, ignore_index=None):
    pair_target = mask_batch_target(target, masked_tokens, vocab_size)
    lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    correct = torch.sum(lprobs.argmax(1) == pair_target)
    nll_loss = F.nll_loss(lprobs, pair_target, ignore_index=ignore_index, reduction='mean')
    return nll_loss, correct


@register_criterion('rna_legacy')
class RNALegacyCriterion(FairseqCriterion):

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward_tape_task(self, model, sample):
        extra_only = True
        masked_only = False
        # TASK2Criterion_MAP = {}
        src_tokens = sample['input_ids'].detach().cpu().numpy()
        masked_tokens = sample['input_mask'].detach().cpu().numpy()

        _, extra = model(src_tokens=sample['input_ids'],
                            extra_only=extra_only, masked_only=masked_only)  # , masked_tokens=sample['input_mask'])
        sample_size = float(len(extra['pooled_output']))

        if 'xcontact_' in self.task.args.eval_task or 'x2contact_' in self.task.args.eval_task:
            model.encoder.use_checkpoint = True
            tape_loss, correct, total, pred_logits = model.classification_heads[self.task.args.eval_task](
                extra['x_pair'], **sample)
        elif self.task.args.eval_task == 'secondary_structure' or 'contact_' in self.task.args.eval_task:
            sequence_output = extra['inner_states'][-1].transpose(0, 1)
            tape_loss, correct, total, pred_logits = model.classification_heads[self.task.args.eval_task](
                sequence_output, **sample)
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

        # if self.task.args.eval_task == 'contact_prediction':
        if 'contact_' in self.task.args.eval_task:
            aupr = pred_logits * sample_size
            logging_output.update(
                {
                    'cmap_aupr': aupr,
                }
            )

        # the Trainer would do: (sum of local gradient sums from each GPU) / (sum of sample_size across GPUs)
        returned_sample_size = sample_size
        return tape_loss, returned_sample_size, logging_output
    
    def foward_pretraining(self, model, sample, reduce):
        # with or without masked tokens input, which means include MLM/PSSM Pred. as top tasks
        tasks = self.task.args.pretraining_tasks.split(',')
        with_masked_input_flag = any(item in tasks for item in ['mlm', 'pmlm', 'pssm'])
        with_unmasked_input_flag = any(item in tasks for item in ['ssa', 'dca', 'cmap'])
        logging_output = {}

        loss = torch.Tensor([0]).to(sample['target'].device)

        if with_masked_input_flag:
            extra_only = False
            masked_only = True
            masked_tokens = sample['target'].ne(self.padding_idx)
            mlm_sample_size = masked_tokens.int().sum()
            pmlm_sample_size = (masked_tokens.int().sum(-1)**2).sum()

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

            logits, pair_logits, _ = model(**sample['net_input'], masked_tokens=masked_tokens, masked_only=masked_only,
                                           extra_only=extra_only)

            del _

            if self.task.args.mlm_loss and 'mlm' in self.task.args.pretraining_tasks or 'pmlm' in self.task.args.pretraining_tasks:
                mlm_loss, mlm_correct = _cross_entropy_pytorch(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=self.padding_idx,
                )
                mlm_loss = mlm_loss * sample['nsentences']

                pmlm_loss, pmlm_correct = _pair_loss(
                    pair_logits,
                    targets.view(-1), masked_tokens, logits.size(-1),
                    ignore_index=self.padding_idx,
                )
                pmlm_loss = pmlm_loss * sample['nsentences']

            if self.task.args.pssm_loss and 'pssm' in self.task.args.pretraining_tasks:
                if masked_tokens is not None:
                    sample['pssm_matrix'] = sample['pssm_matrix'][masked_tokens]

                # TODO 4, 24 are hard coded AA's token numbers for calculate pssm loss
                if self.task.args.pssm_loss_fn == 'topk':
                    topk_num = 5  # hard coded
                    values, indices = torch.topk(sample['pssm_matrix'], topk_num, dim=-1)
                    pred_logits = torch.index_select(logits[:, 4:24], -1, indices[0])
                    target_logits = torch.index_select(sample['pssm_matrix'], -1, indices[0])
                    pssm_loss = l2_loss(pred_logits, target_logits)
                    _, pre_indices = torch.topk(logits[:, 4:24], topk_num, dim=-1)
                    coveraged_num = sum(
                        [sum([aa in indices[index] for aa in sample]) for index, sample in enumerate(pre_indices)])
                    pssm_total_nums = len(pre_indices) * topk_num

                    del _
                    del pred_logits
                    del target_logits
                elif self.task.args.pssm_loss_fn == 'kl_loss':
                    eps = 1e-7
                    pssm_loss = kl_loss(torch.log(F.softmax(logits[:, 4:24], dim=-1) + eps), sample['pssm_matrix'])
                elif self.task.args.pssm_loss_fn == 'l1_loss':
                    pssm_loss = l1_loss(logits[:, 4:24], sample['pssm_matrix'])
                else:
                    raise Exception("self.task.args.pssm_loss_fn must be one of topk, kl_loss or l1_loss")

                pssm_loss = pssm_loss * sample['nsentences']

                if self.task.args.mlm_loss and 'mlm' in self.task.args.pretraining_tasks:
                    loss = pssm_loss * self.task.args.pssm_loss + mlm_loss * self.task.args.mlm_loss
                    logging_output.update({
                        'mlm_loss': (mlm_loss * self.task.args.mlm_loss).data,
                        'pssm_loss': (
                                pssm_loss * self.task.args.pssm_loss).data,
                        'pssm_coveraged_num': float(coveraged_num * sample['nsentences']),
                        'pssm_total': float(pssm_total_nums * sample['nsentences'])
                    })
                else:
                    # only pssm situation, also need to calculate correct number
                    loss = pssm_loss * self.task.args.pssm_loss
                    lprobs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32)
                    mlm_correct = torch.sum(lprobs.argmax(1) == targets.view(-1))
                    logging_output.update(
                        {'pssm_loss': (pssm_loss * self.task.args.pssm_loss).data}
                    )
                    del lprobs
            else:
                # loss = mlm_loss * self.task.args.mlm_loss
                loss = pmlm_loss + mlm_loss
                logging_output.update(
                    {
                        'mlm_loss': mlm_loss.data,
                        'pmlm_loss': pmlm_loss.data
                    }
                )

            # acc_mlm is criterion for both mlm & pssm pred.
            logging_output.update(
                {
                    'mlm_correct': utils.item(mlm_correct.data),
                    'pmlm_correct': utils.item(pmlm_correct.data),
                    'mlm_sample_size': mlm_sample_size,
                    'pmlm_sample_size': pmlm_sample_size,
                }
            )

            del logits

        # tasks: dca,cmap,ssa
        if with_unmasked_input_flag:
            extra_only = True
            masked_only = False
            # no need to pass src_lengths parameter to the model
            logits, extra = model(src_tokens=sample['unmasked_src_tokens'],
                                    # src_lengths=sample['net_input']['src_lengths'],
                                    masked_tokens=None, extra_only=extra_only, masked_only=masked_only)
            del logits
            # sequence_output = extra['inner_states'][-1].transpose(0, 1)
            if 'cmap' in tasks:
                contact_loss, cmap_correct, cmap_total, aupr = \
                    model.classification_heads['cmap'](
                        extra['inner_states'][-1].transpose(0, 1),
                        protein_length=sample['net_input']['src_lengths'],
                        targets=sample['contact_map'].long()
                    )
                contact_loss = contact_loss * sample['nsentences']
                aupr = aupr * sample['nsentences']

                loss += contact_loss * self.task.args.cmap_loss
                logging_output.update(
                    {'contact_loss': (contact_loss * self.task.args.cmap_loss).data,
                        'cmap_correct': float(cmap_correct),
                        'cmap_sample_size': float(cmap_total),
                        'cmap_aupr': aupr,
                        # 'cmap_true_labels': true_labels,
                        # 'cmap_predicted_probs': predicted_probs,
                        }
                )

            if 'ssa' in tasks:
                assert len(sample['ssa_matrix']) > 0, \
                    "error with SSA's paired scheme, set max tokens to include at least 2 sequences"
                if len(sample['ssa_matrix']) > 0:
                    ssa_loss, ssa_correct, ssa_total, ssa_mse, (ssa_y_hat, ssa_y) = \
                        model.classification_heads['ssa'](
                            extra['inner_states'][-1].transpose(0, 1), sample['ssa_matrix']
                        )
                    ssa_loss = ssa_loss * sample['nsentences']

                    loss += ssa_loss * self.task.args.ssa_loss

                else:  # suppose this would'n appear
                    ssa_loss = torch.Tensor([0]).to(loss.device)
                    ssa_correct = torch.Tensor([0]).to(loss.device)
                    ssa_total = 0

                logging_output.update(
                    {'ssa_loss': (ssa_loss * self.task.args.ssa_loss).data,  # return a sum loss instead
                        'ssa_correct': utils.item(ssa_correct.data),
                        'ssa_sample_size': ssa_total,
                        'ssa_y_hat': ssa_y_hat,
                        'ssa_y': ssa_y
                        }
                )

        logging_output.update(
            {
                'loss': loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['nsentences'],
            }
        )

        # the Trainer would do: (sum of local gradient sums from each GPU) / (sum of sample_size across GPUs)
        returned_sample_size = sample['nsentences']
        return loss, returned_sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        The pipeline is that:
        1. based on the incoming args, decide which head/s are used
        2. feed the heads input samples and get corresponding loss value
        3. log the output and return the loss

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if self.task.args.eval_task:
            return self.forward_tape_task(model, sample)
        else:
            return self.foward_pretraining(model, sample, reduce)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training.
        Use a flag key-word for choosing which training/evalution task
        """

        nsentences = float(sum(log['nsentences'] for log in logging_outputs))

        if 'tape_loss' in logging_outputs[0]:  # TAPE's: fluorescence, stability
            # sample_size here is the tokens/units's num for calculate (mean) loss
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
        else:
            """loss/*_loss is the logged loss value over (num_)updates, corresponding to args.log_interval
            sample_size here is the number of total masked tokens
            ppl is the 2**loss, here the loss is the 'mean' loss for each tokens
            
            meter is the center log component for stroring values
            metrics's log_scalar/log_derived contributes to the meter's logged value
            """

            loss_sum = sum(log.get('loss', 0) for log in logging_outputs)  # 'loss' must be logged as required
            metrics.log_scalar('loss', loss_sum / nsentences / math.log(2), weight=nsentences,
                               round=5)  # stored in meter
            
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        SSA/cmap/fl/stability are not suitable for distributed sum
        """
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
