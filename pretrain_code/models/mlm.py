import logging
import os
import numpy as np
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fairseq import utils, checkpoint_utils
from fairseq.checkpoint_utils import load_pretrained_component_from_model
from fairseq.models.fconv import FConvDecoder
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)

from fairseq.models.masked_lm import MaskedLMEncoder

from fairseq.modules import (
    LayerNorm,
    SinusoidalPositionalEmbedding,
    TransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from . import prediction

logger = logging.getLogger(__name__)


@register_model('rna_mlm')
class RNABaseModel(BaseFairseqModel):
    """Base class for encoder-only models for protein.

    Args:
        encoder (FairseqEncoder): the encoder
    """

    def __init__(self, args, encoder, classification_heads=nn.ModuleDict()):
        super().__init__()
        self.encoder = encoder
        self.args = args
        self.classification_heads = classification_heads

        assert isinstance(self.encoder, FairseqEncoder)
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)  # TODO whether triage bug

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--pooler_dropout', type=float, default=0.,
                            help='used in classification head')

        # Arguments related to dropout
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float,
                            metavar='D', help='dropout probability for'
                                              ' attention weights')
        parser.add_argument('--act-dropout', type=float,
                            metavar='D', help='dropout probability after'
                                              ' activation in FFN')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input'
                                                      ' and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings'
                                 ' (outside self attention)')
        parser.add_argument('--num-segment', type=int, metavar='N',
                            help='num segment in the input')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set,'
                                                                     ' calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

        # misc params
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            # "relu","gelu","gelu_accurate","tanh","linear"
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for pooler layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')

        parser.add_argument('--pretrained-file', type=str, default=None,
                            help='path to load the previous encoder')
        parser.add_argument('--label-padding', type=int, default=-1,
                            help='used for decided which token for padding labels.')

    def forward(self, src_tokens, segment_labels=None, extra_only=False, masked_only=False,
                classification_head_name=None, **kwargs):
        """
        Run the forward pass for a encoder-only model.

        Feeds a batch of tokens through the encoder to generate features.

        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the encoder's output, typically of shape `(batch, src_len, features)`
        """
        x, extra = self.encoder(src_tokens, segment_labels=segment_labels,
                                extra_only=False, masked_only=False, **kwargs)
        
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions

    def register_classification_head(self, name, **kwargs):
        """Register a classification head."""

        if name in self.classification_heads:
            logger.warning('Using registered head "{}"'.format(name))
        else:
            self.classification_heads[name] = prediction.DOWNSTREAM_HEADS[name](self.args)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        This model is 1) pre-trained for a transformer-encoder
        2) fine-tuning trained for various evaluation tasks include TAPE's 5 tasks
        3) capable of embedding & validating & testing & evaluation
        """

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        if args.pretrained_file:
            """
            Arguments:
                state_dict (dict): a dict containing parameters and
                    persistent buffers.
                strict (bool, optional): whether to strictly enforce that the keys
                    in :attr:`state_dict` match the keys returned by this module's
                    :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            make sure all arguments are present in older models
            """
            models, pharsed_args, _ = checkpoint_utils.load_model_ensemble_and_task(
                args.pretrained_file.split(os.pathsep),
                arg_overrides={
                    'data': getattr(args, "data", None), 
                    'eval_task': None,  # legacy
                    'max_positions': args.max_positions, 
                    'tokens_per_sample': args.tokens_per_sample,
                    'task': args.task,
                    'arch': args.arch,
                    'criterion': args.criterion,
                    '_name': None, # For compatibility
                    'eval_task': args.eval_task
                },
                suffix=getattr(args, "checkpoint_suffix", ""),
                task=task,
                strict=False,  # TODO
            )
            model = models[0]
            model.encoder.remove_head()
            print('Loaded pre-trained model encoder from ', args.pretrained_file.split(os.pathsep))
            logger.info(args)
            return cls(args, model.encoder)

        base_architecture(args)

        logger.info(args)

        encoder = RNAMaskedLMEncoder(args, task.dictionary)

        return cls(args, encoder)


class RNAMaskedLMEncoder(MaskedLMEncoder):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_positions = args.max_positions

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None

        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.pooler_activation = utils.get_activation_fn(args.pooler_activation_fn)

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(self.vocab_size))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, self.vocab_size, bias=False
                )

            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    args.encoder_embed_dim, self.sentence_out_dim, bias=False
                )

    def remove_head(self):
        self.lm_output_learned_bias = None
        self.embed_out = None

    def forward(self, src_tokens, segment_labels=None, masked_tokens=None,
                extra_only=False, masked_only=False, **unused):
        """
        Args:
            - src_tokens: B x T matrix representing sentences
            - segment_labels: B x T matrix representing segment label for tokens
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'pooled_output' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """

        inner_states, sentence_rep = self.sentence_encoder(
            src_tokens,
            segment_labels=segment_labels,
        ) # inner_states[-1]: (T, B, C)

        if extra_only:
            x = None
        else:
            x = inner_states[-1].transpose(0, 1)  # (T, B, C) -> (B, T, C)

            sentence_logits = x[0, :, :]

            # project masked tokens only
            if masked_tokens is not None:  # masked_tokens: (B, T)
                x = x[masked_tokens, :]

            x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

            if self.share_input_output_embed and hasattr(
                    self.sentence_encoder.embed_tokens, "weight"
            ):
                x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
            elif self.embed_out is not None:
                x = self.embed_out(x)
                
            if self.lm_output_learned_bias is not None:
                x = x + self.lm_output_learned_bias
            
        if not masked_only:
            pooled_output = self.pooler_activation(self.masked_lm_pooler(sentence_rep))
        else:
            del inner_states
            inner_states = None
            pooled_output = None  # for saving GPU storage and calculation time

        return x, {
            "inner_states": inner_states,
            "pooled_output": pooled_output,
            "sentence_logits": sentence_logits,
        }

    def max_positions(self):
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(
                self.sentence_encoder.embed_positions, SinusoidalPositionalEmbedding
        ):
            state_dict[name + ".sentence_encoder.embed_positions._float_tensor"] \
                = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                        "embed_out.weight" in k or 
                        "sentence_projection_layer.weight" in k or 
                        "lm_output_learned_bias" in k
                ):
                    del state_dict[k]
        return state_dict


@register_model_architecture('rna_mlm', 'rna_mlm_base')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)

    args.share_encoder_input_output_embed = getattr(
        args, 'share_encoder_input_output_embed', True)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.num_segment = getattr(args, 'num_segment', 2)

    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.sent_loss = getattr(args, 'sent_loss', True)

    args.apply_bert_init = getattr(args, 'apply_bert_init', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)

@register_model_architecture('rna_mlm', 'rna_mlm_small')
def prot_small_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    base_architecture(args)

@register_model_architecture('rna_mlm', 'rna_mlm')
def prot_architecture(args):
    base_architecture(args)

@register_model_architecture("rna_mlm", "rna_mlm_big")
def prot_big_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_layers = getattr(args, "encoder_layers", 36)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 5120)
    base_architecture(args)

@register_model_architecture("rna_mlm", "rna_mlm_large")
def prot_large_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", 34)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    base_architecture(args)

@register_model_architecture("rna_mlm", "rna_mlm_xl")
def prot_xl_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", 34)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    base_architecture(args)

@register_model_architecture("rna_mlm", "rna_mlm_1b")
def prot_1b_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_layers = getattr(args, "encoder_layers", 34)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 20)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 5120)
    base_architecture(args)

@register_model_architecture('rna_mlm', 'rna_mlm_xbase')
def xbase_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)

    args.share_encoder_input_output_embed = getattr(
        args, 'share_encoder_input_output_embed', True)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.num_segment = getattr(args, 'num_segment', 2)

    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.sent_loss = getattr(args, 'sent_loss', True)

    args.apply_bert_init = getattr(args, 'apply_bert_init', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)

@register_model_architecture("rna_mlm", "rna_mlm_finetune")
def prot_finetune_architecture(args):
    import warnings
    warnings.filterwarnings(
        action='ignore',
        module='fairseq.utils'
    )