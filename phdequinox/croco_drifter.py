import os
from glob import glob
#import threading

import numpy as np
import dask.dataframe as dd
#from dask import delayed
import pandas as pd
import xarray as xr
from scipy import signal

#%matplotlib inline
#from matplotlib import pyplot as plt
#import matplotlib.animation as anima


class drifter_dataframe(object):

    def __init__(self, run_path, 
                 parquet=True,
                 tdir_max=0, 
                 persist=True):
        """ Description ...
        
        Parameters
        ----------
        ...
        """
        self.run_path = run_path
        df = self._load_data(parquet, tdir_max)
        #
        dxy = int(self.run_path[run_path.find('km')-1])
        df['x'] = df.xgrid*dxy
        df['y'] = df.ygrid*dxy
        self.df = df
        self.dxy = dxy
        #
        if persist:
            self.df = self.df.persist()
            
    def __repr__(self):
        return str(self.df.head())

    def _load_data(self, parquet, tdir_max):
        """ load data into a dask dataframe
        
        Parameters
        ----------
        parquet: boolean
            Activates parquet file reading if present
        tdir_max: int
            Limits number of tdir considered when reading raw text files
        """
        self.parquet_path = os.path.join(self.run_path, 
                                         'diagnostics/floats.parquet')        
        # test if parquet
        if parquet and os.path.isdir(self.parquet_path):
            return dd.read_parquet(self.parquet_path,
                                   engine='fastparquet')
        else:
            return self._load_txt(tdir_max)
                
    def _load_txt(self, tdir_max):
        """ Load original text files
        """
        if tdir_max==0:
            t='t?'
        else:
            t='t[1-%d]'%tdir_max
        df = dd.read_csv(glob(self.run_path+t+'/float.????'),
                 names=['id','time','xgrid','ygrid','zgrid',
                        'depth','temp',
                        'u','v','dudt','dvdt',
                        'pres'],
                 delim_whitespace=True)
        return df
    
    def store_parquet(self, partition_size='100MB'):
        """ store data under parquet format
<<<<<<< HEAD
=======

>>>>>>> 8e7cd93fac8a6280521ab64bcc861cd8cd41ed4e
        Note: could shuffle data by float id here ...
        https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.DataFrame.set_index
        df = df.set_index('id')
        ...
        
        Parameters
        ----------
        partition_size: str, optional
            size of each partition that will be enforced
            Default is '100MB' which is dask recommended size
        """
        # check diagnostic dir exists
        _dir = os.path.join(self.run_path, 'diagnostics')
        if not os.path.isdir(_dir):
            os.mkdir(_dir)
        #
        df = self.df
        # repartition such that each partition is 100MB big
        if partition_size:
            df = df.repartition(partition_size=partition_size)
        #
        df.to_parquet(self.parquet_path, engine='fastparquet')
    
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

<<<<<<< HEAD
def correlate(v1, v2, N, detrend = False, dt=None):
    ''' Compute a lagged correlation between two time series
    These time series are assumed to be regularly sampled in time 
    and along the same time line.
    
    Parameters
    ----------
    
        v1, v2: ndarray, pd.Series
            Time series to correlate, the index must be time if dt is not Provided
            
        N: int
            Length of the output
            
        dt: float, optional
            Time step
            
        detrend: boolean, optional
            Turns detrending on or off. Default is False.

    See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    '''
    if dt is None:
        dt = v1.reset_index()['index'].diff().mean()
    
    if v1 is None and v2 is None:
        _v1 = np.random.randn(N*2)
        _v2 = np.random.randn(N*2)
        if detrend:
            pass
        vv = np.correlate(_v1, _v2, mode='same')
    else:
        if detrend:
            v1 = signal.detrend(v1)
            v2 = signal.detrend(v2)
        
        #print('!!! Not implemented yet')
        # https://www.machinelearningplus.com/time-series/time-series-analysis-python/
        
        vv = np.correlate(v1, v2, mode='same')
    return pd.Series(vv[int(vv.size/2):][:N], index=np.arange(N)*dt)
=======
def _check_directory(dir, create=False):
    """ Check existence of a directory and create it if necessary
    """
    _dir = path.join(run_dir, '')
    # create diagnostics dir if not present
    if path.isdir(directory):
        # directory is an absolute path
        _dir = directory
    elif path.isdir(path.join(dirname, directory)):
        # directory is relative
        _dir = path.join(dirname, directory)
    else:
        if create:
            # need to create the directory
            _dir = path.join(dirname, directory)
            os.mkdir(_dir)
            print('Create new diagnostic directory {}'.format(_dir))
        else:
            raise OSError('Directory does not exist')
    return _dir
>>>>>>> 8e7cd93fac8a6280521ab64bcc861cd8cd41ed4e
