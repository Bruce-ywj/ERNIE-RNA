import typing
import math

import numpy as np
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from torch.utils.checkpoint import checkpoint

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_fn(name: str) -> typing.Callable:
    if name == 'gelu':
        return gelu
    elif name == 'relu':
        return torch.nn.functional.relu
    elif name == 'swish':
        return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")


class MLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)


class ConvNet(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_dim),  # Added this
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x


def accuracy(logits, labels, ignore_index: int = -1):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


def correct_total(logits, labels, ignore_index: int = -1):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return int(correct.sum().item()), int(valid_mask.sum().item())


class Accuracy(nn.Module):

    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target, self.ignore_index)


class CorrectTotal(nn.Module):

    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return correct_total(inputs, target, self.ignore_index)


class SoftSymmetricAlignmentHead(nn.Module):

    def __init__(self, args, bias_init=10):
        super().__init__()
        # default num_classes=class-1
        # ordinal classification
        # initialize ordinal regression bias with 10 following ICLR19_Bepler
        num_classes = 4  # getattr(args, "num_classes", 4)
        self.ordinal_weight = nn.Parameter(torch.randn(1, num_classes))  # correct from bepler's not inited code
        self.ordinal_bias = nn.Parameter(torch.randn(num_classes) + bias_init)

    def forward(self, seqs_logits: list, targets=None, **unused):
        # get similarity logits
        half_batch_len = len(seqs_logits) // 2
        logits = list(
            map(self.score_plus, seqs_logits[:half_batch_len], seqs_logits[half_batch_len:2 * half_batch_len]))
        logits = torch.stack(logits, 0)

        if targets is not None:
            targets = Variable(targets)

            ssa_loss = F.binary_cross_entropy_with_logits(logits, targets.float())
            # correct = sum(torch.prod(torch.round(logits).long() == targets.long(), -1))

            with torch.no_grad():

                p = torch.sigmoid(logits)
                ones = p.new(half_batch_len, 1).zero_() + 1
                p_ge = torch.cat([ones, p], 1)
                p_lt = torch.cat([1 - p, ones], 1)
                p = p_ge * p_lt
                p = p / p.sum(1, keepdim=True)  # make sure p is normalized

                _, y_hard = torch.max(p, 1)
                levels = torch.arange(5).to(p.device)
                y_hat = torch.sum(p * levels.float(), 1)
                y = torch.sum(targets.data, 1)

                # loss = F.cross_entropy(p, y).item()  # calculate cross entropy loss from p vector

                correct = torch.sum((y == y_hard).float())
                mse = torch.mean((y.float() - y_hat) ** 2)

            return ssa_loss, correct, len(logits), mse, (y_hat.cpu().numpy(), y.cpu().numpy())
        else:
            return None, None, None, logits

    def score_plus(self, z0, z1):
        # compute similarity score with soft-alignment
        # this could be used for computing a in/out msa score
        s = -torch.sum(torch.abs(z0.unsqueeze(1) - z1), -1)
        a, b = F.softmax(s, 1), F.softmax(s, 0)
        c = a + b - a * b
        c = torch.sum(c * s) / torch.sum(c)
        logit = c * self.ordinal_weight + self.ordinal_bias
        return logit.view(-1)


class ValuePredictionHead(nn.Module):
    """
    For regression tasks: TAPE fluorescence & stability
    """

    def __init__(self, args, dropout: float = 0.):
        super().__init__()
        hidden_size = getattr(args, "encoder_embed_dim", 512)

        hidden_dimention = 512
        self.value_prediction = MLP(hidden_size, hidden_dimention, 1, dropout)

    def forward(self, pooled_output, targets=None, **unused):
        value_pred = self.value_prediction(pooled_output)
        # outputs = (value_pred,)

        if targets is not None:
            loss_fct = nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            # outputs = (value_pred_loss, None, None) + outputs

        return value_pred_loss, None, None, value_pred  # (loss), value_prediction


class SequenceClassificationHead(nn.Module):

    def __init__(self, args):
        super().__init__()
        hidden_size = getattr(args, "encoder_embed_dim", 512)
        num_labels = 1195  # getattr(args, "num_classes", 1195) # Remote homology prediction
        self.classify = MLP(hidden_size, hidden_size, num_labels)

    def forward(self, pooled_output, targets=None, **unused):
        logits = self.classify(pooled_output)
        outputs = (logits,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, targets)
            correct, total = correct_total(logits, targets)
            # metrics = {'accuracy': accuracy(logits, targets)}
            # loss_and_metrics = (classification_loss, metrics)

            # loss_and_metrics = (classification_loss, correct, total)
            # outputs = (loss_and_metrics,) + outputs

        # returned correct, total are int type
        return classification_loss, correct, total, outputs  # (loss), logits


class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self, args):
        super().__init__()

        hidden_size = getattr(args, "encoder_embed_dim", 512)
        num_labels = getattr(args, "num_classes", 3)  # Secondary structure prediction
        ignore_index = getattr(args, "label_padding", -1)

        self.classify = ConvNet(hidden_size, 512, num_labels)
        self.num_labels = num_labels
        self._ignore_index = ignore_index

    def forward(self, sequence_output, targets=None, **unused):
        sequence_logits = self.classify(sequence_output)
        outputs = (sequence_logits,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            classification_loss = loss_fct(
                sequence_logits.view(-1, self.num_labels), targets.view(-1))
            acc_fct = CorrectTotal(ignore_index=self._ignore_index)
            correct, total = acc_fct(sequence_logits.view(-1, self.num_labels), targets.view(-1))
            # loss_and_metrics = (classification_loss, correct, total)
            # outputs = (loss_and_metrics,) + outputs

        # returned correct, total are int type
        return classification_loss, correct, total, outputs  # (loss), sequence_logits

