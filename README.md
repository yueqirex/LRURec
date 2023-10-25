# Linear Recurrent Units for Sequential Recommendation

This repository is the PyTorch impelementation for WSDM 2024 paper:

**Linear Recurrent Units for Sequential Recommendation [[Paper](https://browse.arxiv.org/pdf/2310.02367.pdf)][[Code](https://github.com/yueqirex/lrurec)]** (BibTex citation at the bottom)

Zhenrui Yue*, Yueqi Wang*, Zhankui He†, Huimin Zeng, Julian McAuley, Dong Wang. Linear Recurrent Units for Sequential Recommendation.

<img src=media/overall_model_arch.png width=1000>


## Requirements

Numpy, pandas, pytorch etc. For our detailed running environment see requirements.txt


## How to run LRURec
The command below specifies the training of LRURec on MovieLens-1M.
```bash
python train.py --dataset_code=ml-1m
```

Excecute the above command (with arguments) to train LRURec, select dataset_code from ml-1m, beauty, video, sports, steam and xlong. XLong must be downloaded separately and put under ./data/xlong for experiments. Once trainin is finished, evaluation is automatically performed with models and results saved in ./experiments.


## Performance

The table below reports our main performance results, with best results marked in bold and second best results underlined. For training and evaluation details, please refer to our paper.

<img src=media/performance.png width=1000>


## Citation
Please consider citing the following paper if you use our methods in your research:
```bib
@article{yue2023linear,
  title={Linear Recurrent Units for Sequential Recommendation},
  author={Yue, Zhenrui and Wang, Yueqi and He, Zhankui and Zeng, Huimin and McAuley, Julian and Wang, Dong},
  journal={arXiv preprint arXiv:2310.02367},
  year={2023}
}
```
