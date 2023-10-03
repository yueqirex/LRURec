import torch
import wandb
import argparse

from pytorch_lightning import seed_everything
from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *


def train(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code + \
            '_' + str(args.weight_decay) + '_' + str(args.bert_dropout) + '_' + str(args.bert_attn_dropout)

    train, val, test = dataloader_factory(args)
    model = LRU(args)
    trainer = LRUTrainer(args, model, train, val, test, export_root, args.use_wandb)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    set_template(args)
    train(args)
