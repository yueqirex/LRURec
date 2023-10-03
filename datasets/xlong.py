from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
tqdm.pandas()


class XLongDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'xlong'

    @classmethod
    def url(cls):
        return None  # download train_corpus_total_dual.txt & test_corpus_total_dual.txt

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['train_corpus_total_dual.txt', 'test_corpus_total_dual.txt']

    def maybe_download_raw_dataset(self):
        pass
    
    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = df.groupby('uid').progress_apply(lambda d: list(d['sid']))
            train, val, test = {}, {}, {}
            for i in range(user_count):
                user = i + 1
                items = user2items[user]
                if len(items) < 3:
                    train[user], val[user], test[user] = items, [], []
                else:
                    train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.remove_immediate_repeats(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        train, val
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        def split_sessions(seq):
            seq = [int(x.strip()) for x in seq.split(",")]
            return seq
        
        folder_path = self._get_rawdata_folder_path()
        train_path = folder_path.joinpath('train_corpus_total_dual.txt')
        test_path = folder_path.joinpath('test_corpus_total_dual.txt')
        train = pd.read_csv(train_path, header=None, sep='\t')
        test = pd.read_csv(test_path, header=None, sep='\t')
        df = pd.DataFrame(np.concatenate([train, test], axis=0))

        df[2] = df[2].apply(split_sessions)
        df.apply(lambda x: x[2].append(x[3]), axis=1)

        df = df[2].values
        df = pd.DataFrame(zip(range(len(df)), df), columns=['uid', 'sid'])
        df = df.explode('sid')
        return df
