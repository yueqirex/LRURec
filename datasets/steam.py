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
tqdm.pandas()


class SteamDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'steam'

    @classmethod
    def url(cls):
        return None  # use the provided Steam.txt

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Steam.txt']

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
        # df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Steam.txt')
        df = pd.read_csv(file_path, header=None, sep=' ')
        df.columns = ['uid', 'sid']
        return df
