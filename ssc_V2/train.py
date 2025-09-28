# flake8: noqa
import os.path as osp

from ssc_V2.archs import *
from ssc_V2.data import *
from ssc_V2.models import *

from ssc_V2.train_jscc import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
