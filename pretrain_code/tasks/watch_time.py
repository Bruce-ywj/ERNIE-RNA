from mlm import *
from fairseq import checkpoint_utils, options, tasks, utils, data, options
import os
import sys
sys.path.append("..")
from models.mlm import *
sys.path.append("..")
from criterions.mlm import *
from fairseq.data import (
    data_utils,
    Dictionary,
    iterators,
    FairseqDataset,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
    MaskTokensDataset,
    PadDataset,
    BaseWrapperDataset,
)
from fairseq.data import Dictionary, data_utils, BaseWrapperDataset, LRUCacheDataset
from functools import lru_cache
import math


mlm_pretrained_model = '../../../../rna-emb/ITP-RNAcentral-cdhit-rna_mlm_base-MAXLEN1024-ckpt/checkpoint38.pt'
arg_overrides = { "data": '../../../../../../data/RNAcentral_data/rnacentral_data_cdhit_ftvocab_bin/' }

models, args, task = checkpoint_utils.load_model_ensemble_and_task(mlm_pretrained_model.split(os.pathsep), 
                                                                   arg_overrides=arg_overrides)

epoch = 1
split = 'valid'
paths = utils.split_paths(task.args.data)
assert len(paths) > 0
data_path = paths[(epoch - 1) % len(paths)]
split_path = os.path.join(data_path, split)
dataset = data_utils.load_indexed_dataset(
            split_path,
            task.source_dictionary,
            task.args.dataset_impl,
            combine=False,
        )

dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            task.args.tokens_per_sample - 1,  # one less for <s>
            pad=task.source_dictionary.pad(),
            eos=task.source_dictionary.eos(),
            break_mode=task.args.sample_break_mode,
        )

dataset = PrependTokenDataset(dataset, task.source_dictionary.bos())

def Gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x,y):
    if x == 5 and y == 6:
        return 2
    elif x == 4 and y == 7:
        return 3
    elif x == 4 and y == 6:
        return 0.8
    elif x == 6 and y == 5:
        return 2
    elif x == 7 and y == 4:
        return 3
    elif x == 6 and y == 4:
        return 0.8
    else:
        return 0

def creatmat(data):
    mat = np.zeros([len(data),len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add <len(data):
                    score = paired(data[i - add],data[j + add])
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1,30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(data[i + add],data[j - add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i],[j]] = coefficient
    return mat

class MaskTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.
    Input items are masked according to the specified masking probability.
    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, two_dim_score=True, two_dim_mask=-1)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        two_dim_score: bool = False,
        two_dim_mask: int = -1,
        mask_whole_words: torch.Tensor = None,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.two_dim_score = two_dim_score
        self.two_dim_mask = two_dim_mask
        self.mask_whole_words = mask_whole_words

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[: self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))

            # decide elements to mask
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz
                + np.random.rand()
            )
            mask[np.random.choice(sz, num_mask, replace=False)] = True

            # return 2d-dim socre:
            if self.two_dim_score:
                matrix = creatmat(item)
                # use -1 represent mask
                matrix[mask,:] = self.two_dim_mask
                matrix[:,mask] = self.two_dim_mask
                return torch.from_numpy(matrix)
            
            # return target
            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                if self.mask_whole_words is not None:
                    mask = np.repeat(mask, word_lens)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()

                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )

            return torch.from_numpy(new_item)
        
src_dataset, tgt_dataset, twod_dataset = MaskTokensDataset.apply_mask(
            dataset,
            task.source_dictionary,
            pad_idx=task.source_dictionary.pad(),
            mask_idx=task.mask_idx,
            seed=task.args.seed,
            mask_prob=task.args.mask_prob,
            leave_unmasked_prob=task.args.leave_unmasked_prob,
            random_token_prob=task.args.random_token_prob,
            freq_weighted_replacement=task.args.freq_weighted_replacement,
            mask_whole_words=None,
        )

@profile
def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


samples = [src_dataset[0],src_dataset[1],src_dataset[2],src_dataset[3],src_dataset[4],src_dataset[5]]
oned_input = collate_tokens(samples,1,False)

class PadDataset_2d(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_tokens_2d(samples, self.pad_idx, left_pad=self.left_pad)


@profile
def collate_tokens_2d(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    value_lst = [v.reshape(-1) for v in values]
    size_lst = [v.size(0) for v in values]
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    # res = np.full(shape=(len(values),size,size),fill_value=pad_idx)
    d = torch.arange(0,size).expand(len(values),size,size)
    d = torch.max(d, d.transpose(-1,-2))
    d_mask = d < torch.Tensor(size_lst)[:,None,None]
    # def copy_tensor(src, dst):
    #     assert dst.numel() == src.numel()
    #     if move_eos_to_beginning:
    #         if eos_idx is None:
    #             # if no eos_idx is specified, then use the last token in src
    #             dst[0] = src[-1]
    #         else:
    #             dst[0] = eos_idx
    #         dst[1:] = src[:-1]
    #     else:
    #         dst.copy_(src)

    # for i, v in enumerate(values):
    #     # copy_tensor(v, res[i][size - len(v) :, size - len(v) :] if left_pad else res[i][:len(v),:len(v)])
    #     res[i][:len(v),:len(v)] = v.numpy()
    
    res = torch.ones(len(values),size,size).masked_scatter(d_mask,torch.cat(value_lst))
    return res

samples = [twod_dataset[0],twod_dataset[1],twod_dataset[2],twod_dataset[3],twod_dataset[4],twod_dataset[5]]
twod_input = collate_tokens_2d(samples,1,False)