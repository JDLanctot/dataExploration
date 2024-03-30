import pandas as pd
import matplotlib as mpl
import numpy as np
import random
import os

__all__ = []
__all__.extend([
    'set_seed',
    'set_mpl',
    'import_data'
])

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_mpl() -> None:
    # change defaults to be less ugly for matplotlib
    mpl.rc('xtick', labelsize=14, color="#222222")
    mpl.rc('ytick', labelsize=14, color="#222222")
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    mpl.rc('font', size=16)
    mpl.rc('xtick.major', size=6, width=1)
    mpl.rc('xtick.minor', size=3, width=1)
    mpl.rc('ytick.major', size=6, width=1)
    mpl.rc('ytick.minor', size=3, width=1)
    mpl.rc('axes', linewidth=1, edgecolor="#222222", labelcolor="#222222")
    mpl.rc('text', usetex=False, color="#222222")

def import_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


