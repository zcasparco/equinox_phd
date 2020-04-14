import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.stats import norm
import xarray as xr
import dask.array as da
import dask
from scipy.stats import norm

def generate_data(distrib,*args):
    return distrib(*args)

def mean(values, axis=None):
    """Function to use in bootstrap

    Parameters
    ----------
    values : data to average
    axis : axis or axes along which average
    Returns
    ----------
    mean : ndarray
        Average
    """
    if axis==None:
        return np.mean(values)
    else:
        return np.mean(values, axis=axis)
def resampling(values,B):
    """Creates B samples for the bootstrap (with replacements)

    Parameters
    ----------
    values : ndarray
        data to use
    B : int
        number of samples needed
    Returns
    ----------
    x_resampled : ndarray
        array with dim0=B+1, contains the B samples and the original data in
    the last position of the first axis (x_resampled[-1,:,:])

    """
    if B==1:
        return resample(values,replace=True)
    else:
        x_resampled = np.empty((B+1, values.shape[0],values.shape[1]))
        x_resampled[-1,:,:] = values
        for i in range(B):
            x_resampled[i,:,:] = resample(values, replace=True)
        return x_resampled

def get_percentile(alpha, loc, rms, n):
    """
    Computes the confidence interval of a normal distribution centered around loc.

    Parameters
    ----------
    alpha : float
        percentage wanted for the confidence interval
    loc : float
        true mean of the distribution
    rms : float
        rms of the distribution
    n : int
        sample size
    Returns
    ----------
    _p(alpha) : float
        lower bound of the confidence interval
    _p(1-alpha) : float
        higher bound of the confidence interval
    """
    _p = lambda alpha: norm.ppf(alpha, loc=loc,
                                scale=rms/np.sqrt(n))
    return _p(alpha), _p(1.-alpha)

def bootstrap_delta(b, n, alpha=.05, mean=0., rms=0.1, Nexp=100):
    """
    Parameters

    b : int
        number of samples used for the bootstrap
    n : int
        size of the original data
    alpha : float
        confidence interval wanted
    xmean : float
        mean of the distribution
    xrms : float
        standard deviation of the distribution
    Nexp : int
        number of draws
    Returns

    out : ndarray
        mean and std average over the bootstrap, mean and std of lower band, mean and std of upper band
    """
    x = np.random.normal(mean, rms, size=(n, Nexp))
    sample_mean = x.mean(axis=0)
    X = xr.DataArray(resampling(x, b), dims=['bsample', 'points', 'experiments'])
    deltastar = (X.mean(dim='points')-sample_mean)
    q = deltastar.quantile([alpha,1-alpha], dim='bsample')
    deltal = q.isel(quantile=0)
    deltau = q.isel(quantile=1)
    ci = [sample_mean-deltal,sample_mean-deltau]
#    return np.hstack([sample_mean.mean(), sample_mean.std(),
#                      deltal.mean(),deltal.std(),
#                      deltau.mean(),deltau.std()])
    return np.hstack([deltastar.mean(), deltastar.std(),
                      ci[0].mean(),ci[0].std(),
                      ci[1].mean(),ci[1].std()])
