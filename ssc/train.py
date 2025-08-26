# flake8: noqa
import os.path as osp

from ssc.archs import *
from ssc.data import *
from ssc.models import *

from ssc.train_jscc import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
