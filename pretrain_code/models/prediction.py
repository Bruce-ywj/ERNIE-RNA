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


class PairwiseContactPredictionHead(nn.Module):
    """
    For the contact prediction task
    """

    def __init__(self, args):
        super().__init__()
        hidden_size = getattr(args, "encoder_embed_dim", 768)
        ignore_index = getattr(args, "label_padding", -1)

        self.predict = nn.Sequential(
            nn.Dropout(), nn.Linear(2 * hidden_size, 2))
        self._ignore_index = ignore_index

    def forward(self, inputs, protein_length=None, targets=None, **unused):  # the protein_length counts <sos> & <eos>
        prod = inputs[:, :, None, :] * inputs[:, None, :, :]
        diff = inputs[:, :, None, :] - inputs[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove <sos>/<eos> tokens

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(
                prediction.view(-1, 2), targets.view(-1))
            correct, total, true_labels, predicted_probs = self.compute_correct_total_at_l5(
                protein_length, prediction, targets)
            aupr = sklearn.metrics.average_precision_score(true_labels, predicted_probs)
        
        return contact_loss, correct, total, aupr

    def compute_correct_total_at_l5(self, protein_length, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=protein_length.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)

            correct = 0
            total = 0
            for length, prob, label, mask in zip(protein_length, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()

            true_labels = labels.masked_select(valid_mask.bool()).cpu().numpy()
            predicted_probs = probs.masked_select(valid_mask.bool()).cpu().numpy()

            return int(correct), int(total), np.asarray(true_labels), np.asarray(predicted_probs)

# Short-range: 6-11
# Medium-range: 12-23
# Long-range: >=24
class UnifiedPairwiseContactPredictionHead(nn.Module):
    """
    For the contact prediction task
    """
    # contact_range = ['sml', 'ml', 's', 'm', 'l']
    # top_l_frac    = [1, 2, 5, ...]
    def __init__(self, args): 
        super().__init__()
        hidden_size = getattr(args, "encoder_embed_dim", 768)
        ignore_index = getattr(args, "label_padding", -1)

        self.predict = nn.Sequential(
            nn.Dropout(), nn.Linear(2 * hidden_size, 2))
        self._ignore_index = ignore_index

        self.contact_range = getattr(args, "contact_range", 'ml')
        self.top_l_frac = getattr(args, "top_l_frac", 5)

        if not args.eval_task:
            raise Exception('[eval_task] is not set')

        if args.eval_task == 'contact_esm':
            self.contact_range = 'l'
            self.top_l_frac = 1

        print('---> contact_range:', self.contact_range)
        print('---> top_l_frac:', self.top_l_frac)

    def forward(self, inputs, protein_length=None, targets=None, **unused):  # the protein_length counts <sos> & <eos>
        prod = inputs[:, :, None, :] * inputs[:, None, :, :]
        diff = inputs[:, :, None, :] - inputs[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove <sos>/<eos> tokens

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(
                prediction.view(-1, 2), targets.view(-1))
            correct, total, true_labels, predicted_probs = self.compute_correct_total_at_lp(
                protein_length, prediction, targets)
            try:
                aupr = sklearn.metrics.average_precision_score(true_labels, predicted_probs)
            except:
                aupr = 0.0
        
        return contact_loss, correct, total, aupr

    def compute_correct_total_at_lp(self, protein_length, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=protein_length.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)

            if self.contact_range == 'sml':
                valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            elif self.contact_range == 'ml':
                valid_mask &= ((y_ind - x_ind) >= 12).unsqueeze(0)
            elif self.contact_range == 's':
                valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
                valid_mask &= ((y_ind - x_ind) < 12).unsqueeze(0)
            elif self.contact_range == 'm':
                valid_mask &= ((y_ind - x_ind) >= 12).unsqueeze(0)
                valid_mask &= ((y_ind - x_ind) < 24).unsqueeze(0)
            elif self.contact_range == 'l':
                valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)

            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)

            correct = 0
            total = 0
            for length, prob, label, mask in zip(protein_length, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // self.top_l_frac, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()

            true_labels = labels.masked_select(valid_mask.bool()).cpu().numpy()
            predicted_probs = probs.masked_select(valid_mask.bool()).cpu().numpy()

            return int(correct), int(total), np.asarray(true_labels), np.asarray(predicted_probs)


## Version 1
class CovariationBasedPairwiseContactPredictionHead(nn.Module):
    """
    For the contact prediction task
    """
    # contact_range = ['sml', 'ml', 's', 'm', 'l']
    # top_l_frac    = [1, 2, 5, ...]
    def __init__(self, args): 
        super().__init__()
        hidden_size = getattr(args, "encoder_embed_dim", 768)
        ignore_index = getattr(args, "label_padding", -1)
        vocab_size = getattr(args, "vocab_size", 27)
        # repr_dim = vocab_size * vocab_size
        repr_dim = 2 * hidden_size
        self.predict = nn.Sequential(nn.Dropout(), nn.Linear(repr_dim, 2))
        self._ignore_index = ignore_index
        self.contact_range = getattr(args, "contact_range", 'ml')
        self.top_l_frac = getattr(args, "top_l_frac", 5)
        if not args.eval_task:
            raise Exception('[eval_task] is not set')
        if 'contact_esm' in args.eval_task:
            self.contact_range = 'l'
            self.top_l_frac = 1
        print('---> contact_range:', self.contact_range)
        print('---> top_l_frac:', self.top_l_frac)

    def predict_score(self, x_pair):
        prediction = self.predict(x_pair)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove <sos>/<eos> tokens
        return prediction

    def forward(self, x_pair, protein_length=None, targets=None, **unused):  # the protein_length counts <sos> & <eos>
        # prediction = checkpoint(self.predict_score, x_pair)
        prediction = self.predict_score(x_pair)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(
                prediction.view(-1, 2), targets.view(-1))
            correct, total, true_labels, predicted_probs = self.compute_correct_total_at_lp(
                protein_length, prediction, targets)
            try:
                aupr = sklearn.metrics.average_precision_score(true_labels, predicted_probs)
            except:
                aupr = 0.0
        return contact_loss, correct, total, aupr

    def compute_correct_total_at_lp(self, protein_length, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=protein_length.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            if self.contact_range == 'sml':
                valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            elif self.contact_range == 'ml':
                valid_mask &= ((y_ind - x_ind) >= 12).unsqueeze(0)
            elif self.contact_range == 's':
                valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
                valid_mask &= ((y_ind - x_ind) < 12).unsqueeze(0)
            elif self.contact_range == 'm':
                valid_mask &= ((y_ind - x_ind) >= 12).unsqueeze(0)
                valid_mask &= ((y_ind - x_ind) < 24).unsqueeze(0)
            elif self.contact_range == 'l':
                valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(protein_length, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // self.top_l_frac, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            true_labels = labels.masked_select(valid_mask.bool()).cpu().numpy()
            predicted_probs = probs.masked_select(valid_mask.bool()).cpu().numpy()
            return int(correct), int(total), np.asarray(true_labels), np.asarray(predicted_probs)


### Version 2
class ResConv2dBlock(nn.Module):

    def __init__(self, dmodel, kernel_size):
        super(ResConv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(dmodel, dmodel, kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(dmodel, dmodel, kernel_size, stride=1, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(dmodel)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(identity + self.conv2(x))
        return x

class ResConv2d(nn.Module):
    
    def __init__(self, input_dim, dmodel, nlayers, kernel_size=5):
        super(ResConv2d, self).__init__()
        self.fc = nn.Linear(input_dim, dmodel)
        self.net = nn.Sequential(*[ResConv2dBlock(dmodel, kernel_size) \
            for _ in range(nlayers)])

    def forward(self, x):
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2) # (B, L, L, C) -> (B, C, L, L)
        x = self.net(x)
        x = x.permute(0, 2, 3, 1) # (B, C, L, L) -> (B, L, L, C)
        return x

class CovariationBasedPairwiseContactPredictionHeadV2(nn.Module):

    def __init__(self, args): 
        super().__init__()
        hidden_size = getattr(args, "encoder_embed_dim", 768)
        ignore_index = getattr(args, "label_padding", -1)
        vocab_size = getattr(args, "vocab_size", 27)

        dmodel = getattr(args, "dmodel", 128)
        nlayers = getattr(args, "nlayers", 8)

        # repr_dim = vocab_size * vocab_size
        repr_dim = 2 * hidden_size

        self.predict = nn.Sequential(ResConv2d(repr_dim, dmodel, nlayers), nn.Dropout(), nn.Linear(dmodel, 2))

        self._ignore_index = ignore_index
        self.contact_range = getattr(args, "contact_range", 'ml')
        self.top_l_frac = getattr(args, "top_l_frac", 5)
        if not args.eval_task:
            raise Exception('[eval_task] is not set')
        if 'contact_esm' in args.eval_task:
            self.contact_range = 'l'
            self.top_l_frac = 1
        print('---> contact_range:', self.contact_range)
        print('---> top_l_frac:', self.top_l_frac)

    def predict_score(self, x_pair):
        prediction = self.predict(x_pair)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove <sos>/<eos> tokens
        return prediction

    def forward(self, x_pair, protein_length=None, targets=None, **unused):  # the protein_length counts <sos> & <eos>
        prediction = checkpoint(self.predict_score, x_pair)
        # prediction = self.predict_score(x_pair)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(
                prediction.view(-1, 2), targets.view(-1))
            correct, total, true_labels, predicted_probs = self.compute_correct_total_at_lp(
                protein_length, prediction, targets)
            try:
                aupr = sklearn.metrics.average_precision_score(true_labels, predicted_probs)
            except:
                aupr = 0.0
        return contact_loss, correct, total, aupr

    def compute_correct_total_at_lp(self, protein_length, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=protein_length.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)

            if self.contact_range == 'sml':
                valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            elif self.contact_range == 'ml':
                valid_mask &= ((y_ind - x_ind) >= 12).unsqueeze(0)
            elif self.contact_range == 's':
                valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
                valid_mask &= ((y_ind - x_ind) < 12).unsqueeze(0)
            elif self.contact_range == 'm':
                valid_mask &= ((y_ind - x_ind) >= 12).unsqueeze(0)
                valid_mask &= ((y_ind - x_ind) < 24).unsqueeze(0)
            elif self.contact_range == 'l':
                valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)

            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)

            correct = 0
            total = 0
            for length, prob, label, mask in zip(protein_length, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // self.top_l_frac, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()

            true_labels = labels.masked_select(valid_mask.bool()).cpu().numpy()
            predicted_probs = probs.masked_select(valid_mask.bool()).cpu().numpy()

            return int(correct), int(total), np.asarray(true_labels), np.asarray(predicted_probs)


DOWNSTREAM_HEADS = {
    'ssa': SoftSymmetricAlignmentHead,
    'cmap': PairwiseContactPredictionHead,
    'secondary_structure': SequenceToSequenceClassificationHead,
    'contact_prediction': PairwiseContactPredictionHead,
    'contact_esm': UnifiedPairwiseContactPredictionHead,
    'contact_mlrange': UnifiedPairwiseContactPredictionHead,
    'xcontact_esm': CovariationBasedPairwiseContactPredictionHead,
    'xcontact_mlrange': CovariationBasedPairwiseContactPredictionHead, # CovariationBasedPairwiseContactPredictionHeadV2
    'x2contact_esm': CovariationBasedPairwiseContactPredictionHeadV2,
    'x2contact_mlrange': CovariationBasedPairwiseContactPredictionHeadV2,
    'remote_homology': SequenceClassificationHead,
    'stability': ValuePredictionHead,
    'fluorescence': ValuePredictionHead,
}
