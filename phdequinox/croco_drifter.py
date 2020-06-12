import os
from glob import glob
#import threading
import shutil

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
                 index=None,                 
                 persist=True):
        """ Description ...
        
        Parameters
        ----------
        ...
        """
        self.run_path = run_path
        df = self._load_data(parquet, tdir_max,index)
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

    def _load_data(self, parquet, tdir_max,index=None):
        """ load data into a dask dataframe
        
        Parameters
        ----------
        parquet: boolean
            Activates parquet file reading if present
        tdir_max: int
            Limits number of tdir considered when reading raw text files
        """
        if index:
            self.parquet_path = os.path.join(self.run_path,
                                             'diagnostics/floats_'+index+'.parquet')
        else:
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
    
    def store_parquet(self, 
                      partition_size='100MB', 
                      index=None,
                      overwrite=False,
                     ):
        """ store data under parquet format
        
        Parameters
        ----------
        partition_size: str, optional
            size of each partition that will be enforced
            Default is '100MB' which is dask recommended size
        """
        parquet_path = self.parquet_path
        if index:
            parquet_path = parquet_path.replace('.parquet','_'+index+'.parquet')
        # check diagnostic dir exists
        _dir = os.path.join(self.run_path, 'diagnostics')
        if not os.path.isdir(_dir):
            os.mkdir(_dir)
        # check wether an archive already exists
        if os.path.isdir(parquet_path):
            if overwrite:
                print('deleting existing archive: {}'.format(parquet_path))
                shutil.rmtree(parquet_path)
            else:
                print('Archive already existing: {}'.format(parquet_path))
                return
        #
        df = self.df
        #
        if index:
            df = df.set_index(index)
        # repartition such that each partition is 100MB big
        if partition_size:
            df = df.repartition(partition_size=partition_size)        
        #
        df.to_parquet(parquet_path, engine='fastparquet')
    
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
    if hasattr(df, 'id'):
        dr_id = df.id.unique()[0]
    elif df.index.name=='id':
        dr_id = df.index.unique()[0]
    elif hasattr(df, 'name'):
        # when mapped after groupby
        dr_id = df.name
    else:
        assert False, 'Cannot find float id'
    #
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
        x, y = cdr.mean_position(_p, L)
        # apply myfun
        myfun_out = myfun(*[_p[c] for c in columns], N, **myfun_kwargs)
        # combine with mean position and time
        _out = pd.DataFrame([[x, y]+[dr_id]+list(myfun_out)],
                            columns = index,
                            index = [t+T/2.])
        out.append(_out)
        t+=T*(1-overlap)
    return pd.concat(out)

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
        trend_var_1 = np.nan
        trend_var_2 = np.nan
        vv = np.correlate(_v1, _v2, mode='same')
    else:
        if detrend:
            _v1 = v1
            _v2 = v2
            v1 = signal.detrend(v1)
            v2 = signal.detrend(v2)
            _v1 = _v1-v1
            trend_var_1 = np.mean(_v1**2)
            _v2 = _v2-v2
            trend_var_2 = np.mean(_v2**2)
        #print('!!! Not implemented yet')
        # https://www.machinelearningplus.com/time-series/time-series-analysis-python/
        
        vv = np.correlate(v1, v2, mode='same')
    if detrend :
        out = np.hstack((vv[int(vv.size/2):][:N], np.array([trend_var_1,trend_var_2])))
        index = list(np.arange(N)*dt)+['trend_var_0', 'trend_var_1']
    else :
        out = vv[int(vv.size/2):][:N]
        index=list(np.arange(N)*dt)
    return pd.Series(out,index=index)

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
