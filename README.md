# Linear Recurrent Units for Sequential Recommendation

This repository is the PyTorch impelementation for the paper:

**Linear Recurrent Units for Sequential Recommendation [[Paper](media/paper.pdf)]**

Zhenrui Yue*, Yueqi Wang*, Zhankui He†, Huimin Zeng, Julian McAuley, Dong Wang (2023). Linear Recurrent Units for Sequential Recommendation. arXiv preprint

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
Please consider citing the following papers if you use our methods in your research:
```bib
@article{yue2023linear,
  title={Linear Recurrent Units for Sequential Recommendation},
  author={Zhenrui Yue, Yueqi Wang, Zhankui He, Huimin Zeng, Julian McAuley and Dong Wang},
  journal={arXiv preprint arXiv},
  year={2023}
}
```
