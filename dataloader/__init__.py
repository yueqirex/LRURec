from datasets import dataset_factory
from .sas import *


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = SASDataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
