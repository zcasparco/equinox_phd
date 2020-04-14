import os
from glob import glob
#import threading

import numpy as np
import dask.dataframe as dd
#from dask import delayed
import pandas as pd
import xarray as xr

#%matplotlib inline
#from matplotlib import pyplot as plt
#import matplotlib.animation as anima


class drifter_dataframe(object):

    def __init__(self, run_path, tdir_max=0, persist=True):
        if tdir_max==0:
            t='t?'
        else:
            t='t[1-%d]'%tdir_max
        df = dd.read_csv(glob(run_path+t+'/float.????'),
                 names=['id','time','xgrid','ygrid','zgrid',
                        'depth','temp',
                        'u','v','dudt','dvdt',
                        'pres'],
                 delim_whitespace=True)
        dxy = int(run_path[run_path.find('km')-1])
        df['x'] = df.xgrid*dxy
        df['y'] = df.ygrid*dxy
        if persist:
            df = df.persist()
        self.df = df

    def __repr__(self):
        return str(self.df.head())

    def init_bins(self,**kwargs):
        """
        dr.init_bins(y={'min': 0., 'max': 2800., 'step': 10}, x=...)
        """
        idx = {}
        for key, item in kwargs.items():
            bins = np.arange(item['min'],item['max'], item['step'])
            idx[key] = pd.IntervalIndex.from_breaks(bins)
            self.df[key+'_cut'] = self.df[key].map_partitions(pd.cut, bins=bins)
        self.idx = idx

    def get_stats(self, V, stats, vbin):
        if isinstance(V, str):
            _V = [V]
        else:
            _V = V

        S = []
        for v in _V:
            _ds = (self.df.groupby([vb+'_cut' for vb in vbin])[v].agg(stats)
                 .compute().to_xarray().rename({s: v+'_'+s for s in stats}) #pandas
                )
            S.append(_ds)

        return xr.merge(S).assign_coords(**{vb+'_bins': self.idx[vb].mid for vb in vbin})

    def get_lats(self, stats,vbin):
        ds = (self.df.groupby([vb+'_cut' for vb in vbin])[y].agg(stats)
            .compute().to_xarray().rename({s: 'y'+'_'+s for s in stats}) #pandas
            )

        return ds.assign_coords(**{vb+'_bins': self.idx[vb].mid for vb in vbin})
# should create a method to get default bins

def mean_position(df, L):
    """ compute the mean position, accounts for the wrapping of x
    """
    x = (np.angle(
            np.exp(1j*(df['x']*2.*np.pi/L-np.pi)).mean()
                 ) + np.pi
        )*L/2./np.pi
    y = df['y'].mean()
    return x, y

def latitudes(df):
    """ compute the mean, minimum and maximum latitude (in km)
    """
    ymean,ymin,ymax = df['y'].mean(),df['y'].min(), df['y'].max()
    return ymean,ymin,ymax
