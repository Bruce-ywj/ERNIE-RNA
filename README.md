# ERNIE-RNA

This repository contains codes and pre-trained models for **RNA feature extraction and secondary structure prediction model (ERNIE-RNA)**.
**ERNIE-RNA is superior to the tested RNA feature extraction models (including RNA-FM) in the feature extraction task, and its effect in the secondary structure prediction task is better than RNAfold, UNI-RNA and others.**
You can find more details about **ERNIE-RNA** in our paper, [ERNIE-RNA: An RNA Language Model with Structure-enhanced Representations](https://www.biorxiv.org/content/10.1101/2024.03.17.585376v1)

![Overview](./docs/model.png)

</details>

<details><summary>Table of contents</summary>

- [ERNIE-RNA](#ernie-rna)
  - [Create Environment with Conda ](#create-environment-with-conda-)
  - [Access pre-trained models. ](#access-pre-trained-models-)
  - [Apply ERNIE-RNA with Existing Scripts. ](#apply-ernie-rna-with-existing-scripts-)
    - [1. Embedding Extraction. ](#1-embedding-extraction-)
    - [2. Secondary structure prediction. ](#2-secondary-structure-prediction-)
    - [3. 3D Closeness Prediction ](#3-3d-closeness-prediction-)
  - [Citations ](#citations-)
  - [License ](#license-)

</details>

## Create Environment with Conda 

First, download the repository and create the environment.

```
git clone https://github.com/Bruce-ywj/ERNIE-RNA.git
cd ./ERNIE-RNA
conda env create -f environment.yml
```

Then, activate the "ERNIE-RNA" environment.

```
conda activate ERNIE-RNA
```

## Access pre-trained models. 

There are two subfolders in the model folder, each folder has a link, and you can download the model in the link to the same directory. Or you can download both models from our [drive](https://drive.google.com/drive/folders/10Yz-sdezhmazzVtrtdBGdzzqK1Z6f-Xv)

## Apply ERNIE-RNA with Existing Scripts. 

### 1. Embedding Extraction. 

```
python extract_embedding.py --seqs_path='./data/test_seqs.txt' --device='cuda:0'
```

The model path parameters are set by default and do not need to be changed.

The corresponding feature extraction code is inside this file, and the sequence in the file can be modified when used.

In this file, you can use ERNIE-RNA (twod_mlm) for feature extraction.

Features include cls, tokens, atten_map.

### 2. Secondary structure prediction. 

ERNIE-RNA provides powerful RNA secondary structure prediction capabilities, supporting model parameters from various training datasets and simultaneously providing both fine-tuned model and zero-shot prediction results.

#### Basic Usage:

```bash
python predict_ss_rna.py --dataset_name bpRNA-1m --seqs_path={fasta_dir} --save_path={output_dir} --device=0
```

#### Parameters:

- `--seqs_path`: Path to the FASTA file containing RNA sequences
- `--save_path`: Directory path for output CT files
- `--dataset_name`: RNA structure finetune dataset name, used to automatically select the corresponding model parameter file
- `--ss_rna_checkpoint`: Path to the fine-tuned model parameter file (required when not using `--trainset_name`)
- `--device`: GPU device ID (0, 1, 2...) or 'cpu'

#### Available Dataset Parameter Files:

| Training Dataset         | Model Parameter File                                                        | Application Scenario                                                                                                                                                               |
| ------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bpRNA-1m`             | ERNIE-RNA_attn-map_ss_prediction_bpRNA-1m_checkpoint.pt                     | General RNA structure prediction (bpRNA-1m refered to bpRNA-1m (80))))                                                                                                             |
| `RNAStralign`          | ERNIE-RNA_attn-map_ss_prediction_RNAStralign_checkpoint.pt                  | General RNA structure prediction                                                                                                                                                   |
| `RIVAS`                | ERNIE-RNA_attn-map_ss_prediction_RIVAS_checkpoint.pt                        | Reproduction of RIVAS results                                                                                                                                                     |
| `RNA3DB`               | ERNIE-RNA_attn-map_ss_prediction_RNA3DB_checkpoint.pt                       | Reproduction of RNA3DB-2D results                                                                                                                                                  |
| `bpRNA-new`            | ERNIE-RNA_attn-map_frozen_ss_prediction_bpRNA-1m_checkpoint.pt              | Novel RNA structure prediction (Note: This is the ERNIE-RNA attn-map frozen model trained on the bpRNA-1m dataset, bpRNA-new do not serve as the trainset))                        |
| `bpRNA-1m_RNAstralign` | ERNIE-RNA_attn-map_ss_prediction_bpRNA-1m-all_and_RNAStralign_checkpoint.pt | General RNA structure prediction (Note: Used all bpRNA-1m) and RNAStralign trainset sequences, excluding various(eg. RIVAS, RNA3DB) datasets' valid/test sequences, for training) |

#### Output Files:

For each input sequence, ERNIE-RNA generates two structure files in CT format:

- `{sequence_name}_finetune_prediction.ct`: Prediction results from the model fine-tuned on the specified training dataset
- `{sequence_name}_zeroshot_prediction.ct`: Zero-shot prediction results using the pre-trained model (without fine-tuning)

> **Note**: Regardless of which `dataset_name` is selected, the system will output additional zero-shot prediction results. Zero-shot prediction results have not been fine-tuned on any RNA structure training set and may remain the SAME output regardless of the `dataset_name`.

#### Usage Recommendations:

- For sequences from common Rfam families or RNA families included in the bpRNA-1m and RNAStralign training sets, we recommend using `bpRNA-1m_RNAstralign`, `bpRNA-1m`, or `RNAStralign` parameters
- For sequences that may belong to unknown RNA families, we recommend trying `bpRNA-new` or `RNA3DB` parameters, or referring to the zero-shot prediction results output alongside each finetuned model's predictions

#### Examples:

1. Prediction using bpRNA-1m training set parameters:

```bash
python predict_ss_rna.py --dataset_name bpRNA-1m --device 0 --seqs_path ./data/ss_prediction/bpRNA-1m_testseqs.fasta --save_path ./results/ernie_rna_ss_prediction/bpRNA-1m_test_results/
```

2. Prediction using RNA3DB training set parameters:

```bash
python predict_ss_rna.py --dataset_name RNA3DB --device 0 --seqs_path ./data/ss_prediction/rna3db_testseqs.fasta --save_path ./results/ernie_rna_ss_prediction/rna3db_test_results/
```

3. Prediction using bpRNA-1m training set but performed best in bpRNA-new test parameters:

```bash
python predict_ss_rna.py --dataset_name bpRNA-new --device 0 --seqs_path ./data/ss_prediction/bpRNA-new_testseqs.fasta --save_path ./results/ernie_rna_ss_prediction/bpRNA-new_test_results/
```

### 3. 3D Closeness Prediction 

This section describes how to use ERNIE-RNA to predict RNA 3D closeness maps. This functionality relies on the pre-trained ERNIE-RNA model as a feature extractor and a downstream model head specifically fine-tuned for the 3D closeness task. The recommended downstream model architecture is based on ERNIE-RNA's attention maps.

**Usage Example:**

To predict 3D closeness for RNA sequences in a FASTA file and visualize the results:

```bash
python predict_3d_clossness.py \
    --input_rna_file ./results/ernie_rna_3d_clossness/example.fasta \
    --device cuda:0 \
    --visualize
```

## 4. UTR MRL prediction 

This section describes how to use ERNIE-RNA to predict mean ribosome loading (MRL) for 5' UTR RNA sequences, a key measure of translation efficiency.

### Basic Usage

```bash
python predict_MRL.py \
    --data_roots ./data/MRL_data/seqs.fasta \
    --device 0
```

### Parameters

- `--data_roots`: Path to input FASTA file containing 5'UTR sequences (default: `./data/MRL_data/seqs.fasta`)
- `--bert_path`: Path to ERNIE-RNA pre-trained model checkpoint (default: `./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt`)
- `--model_root`: Path to fine-tuned MRL prediction model weights (default: `./checkpoint/ERNIE-RNA_UTR_MRL_checkpoint/ERNIE-RNA-UTR_ML_CNN_checkpoint.pt`)
- `--scaler_root`: Path to scaler file for normalization (default: `./checkpoint/ERNIE-RNA_UTR_MRL_checkpoint/scaler.save`)
- `--output_dir`: Directory to save prediction results (default: `./results/ernie_rna_utr_mrl`)
- `--device`: GPU device ID to use (default: 0, use -1 for CPU)

## Citations 

If you find the models useful in your research, please cite our work:

[ERNIE-RNA: An RNA Language Model with Structure-enhanced Representations](https://www.biorxiv.org/content/10.1101/2024.03.17.585376v1)

Yin W, Zhang Z, He L, et al. ERNIE-RNA: An RNA Language Model with Structure-enhanced Representations[J]. bioRxiv, 2024: 2024.03. 17.585376.

We use [fairseq](https://github.com/pytorch/fairseq) sequence modeling framework to train our RNA language modeling.
We very appreciate this excellent work!

## License

This source code is licensed under the MIT license found in the `LICENSE` file
in the root directory of this source tree.
