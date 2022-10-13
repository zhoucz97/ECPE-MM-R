# ECPE-MM-R

Data and codes for the COLING2022 paper:  **[A Multi-turn Machine Reading Comprehension Framework with Rethink Mechanism for Emotion-Cause Pair Extraction](https://aclanthology.org/2022.coling-1.584/)**

If you use our codes or your research is related to our paper, please kindly cite our paper:

```
@inproceedings{zhou-etal-2022-multi-turn,
    title = "A Multi-turn Machine Reading Comprehension Framework with Rethink Mechanism for Emotion-Cause Pair Extraction",
    author = "Zhou, Changzhi  and
      Song, Dandan  and
      Xu, Jing  and
      Wu, Zhijing",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.584",
    pages = "6726--6735",
}
```

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

