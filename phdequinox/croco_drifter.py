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
        ds = (self.df.groupby([vb+'_cut' for vb in vbin])['y'].agg(stats)
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

def time_window_processing(df, myfun, columns, T, N, L, overlap=0.5, **myfun_kwargs):
    ''' break each drifter time series into time windows and process each windows
    
    Parameters
    ----------
        
        df: Dataframe
            This dataframe represents a drifter time series
        
        T: float
            Length of the time windows
            
        myfun
            Method that will be applied to each window
            
        columns: list of str
            List of columns of df that will become inputs of myfun
            
        N: int
            Length of myfun outputs
            
        L: int
            Maximum x (used in mean_position)
            
        overlap: float
            Amount of overlap between windows. 
            Should be between 0 and 1. 
            Default is 0.5
            
        **myfun_kwargs
            Keyword arguments for myfun
    
    '''
    try:
        dr_id = df.id.unique()[0]
    except:
        dr_id = df.name
    p = df.sort_values('time').set_index('time')
    tmin, tmax = p.index[0], p.index[-1]
    # need to create an empty dataframe, in case the loop below is empty
    myfun_out = myfun(*[None for c in columns], N, **myfun_kwargs) # get index from fake output
    index = ['x','y']+['id']+list(myfun_out.index)
    out = [pd.DataFrame({_:[] for _ in index})]
    t=tmin
    while t+T<tmax:
        #
        _p = p.loc[t:t+T]
        # compute average position
        x, y = mean_position(_p, L)
        # apply myfun
        myfun_out = myfun(*[_p[c] for c in columns], N, **myfun_kwargs)
        # combine with mean position and time
        _out = pd.DataFrame([[x, y]+[df.id.unique()[0]]+list(myfun_out)],
                            columns = index,
                            index = [t+T/2.])
        out.append(_out)
        t+=T*(1-overlap)
    return pd.concat(out)