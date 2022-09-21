# ECPE-MM-R

This repository contains the code for our COLING 2022 paper:

Changzhi Zhou, Dandan Song, Jing Xu, and Zhijing Wu. **A Multi-turn Machine Reading Comprehension Framework with Rethink Mechanism for Emotion-Cause Pair Extraction** [[pdf](https://arxiv.org/abs/2209.07972)]

Please cite our paper if you use this code.

Some code is based on [BMRC](https://github.com/NKU-IIPLab/BMRC), [Rank-Emotion-Cause](https://github.com/Determined22/Rank-Emotion-Cause), and [DeepInf](https://github.com/xptree/DeepInf).

## Dependencies
- Python==3.7
- PyTorch==1.6
- [Transformers from Hugging Face](https://github.com/huggingface/transformers)

The code has been tested on Ubuntu 18.04 using a single V100 GPU.

## Quick Start

1. Clone or download this repo.
2. Download the pretrained parameters "bert-base-chinese/pytorch_model.bin" from [this link](https://huggingface.co/bert-base-chinese/tree/main). And then put the pytorch_model.bin to the folder `pretrained_model/bert-base-chinese`.
3. Run data preprocessing.
    - `python3 dataProcess.py`
    - `python3 makeData_dual.py`
4. Run our model *MM-R*.
    - `sh run.sh`

