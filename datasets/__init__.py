from .ml_1m import ML1MDataset
from .beauty import BeautyDataset
from .video import VideoDataset
from .sports import SportsDataset
from .steam import SteamDataset
from .xlong import XLongDataset


DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    BeautyDataset.code(): BeautyDataset,
    VideoDataset.code(): VideoDataset,
    SportsDataset.code(): SportsDataset,
    SteamDataset.code(): SteamDataset,
    XLongDataset.code(): XLongDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
