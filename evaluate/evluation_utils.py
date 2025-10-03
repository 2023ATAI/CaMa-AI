import numpy as np
import yaml
import argparse


def getConfig(dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=dir, help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config


def CORRELATION(sim, obs):
    """
        correlation coefficient
        input:
            sim: simulated (N)
            obs: observed (N)
        output:
            correlation: correlation coefficient
        """
    corr = np.corrcoef(sim, obs)[0, 1]

    return corr


def KGE(sim, obs):
    """
           Kling-Gupta Efficiency
           input:
            sim: simulated (N)
            obs: observed (N)
           output:
               kge: Kling-Gupta Efficiency
               cc: correlation
               alpha: ratio of the standard deviation
               beta: ratio of the mean
           """
    cc = CORRELATION(sim, obs)
    alpha = np.std(sim) / np.std(obs)
    # alpha = np.std(s)/np.std(o)
    # beta = s.mean(dim='time') / o.mean(dim='time')
    beta = np.mean(sim) / np.mean(obs)
    kge = 1 - ((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
    return kge


def KGESS(sim, obs):
    """
        Normalized Kling-Gupta Efficiency
        input:
            sim: simulated (N)
            obs: observed (N)
        output:
            kgess:Normalized Kling-Gupta Efficiency
        note:
        KGEbench= −0.41 from Knoben et al., 2019)
        Knoben, W. J. M., Freer, J. E., and Woods, R. A.: Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–
        Gupta efficiency scores, Hydrol. Earth Syst. Sci., 23, 4323–4331,
        https://doi.org/10.5194/hess-23-4323-2019, 2019.
        """
    kge = KGE(sim, obs)
    kgess = (kge - (-0.41)) / (1.0 - (-0.41))
    return kgess  # , cc, alpha, beta


def RMSE(sim, obs):
    """
        Calculate Root Mean Squared Error (RMSE).

        input:
            sim: simulated (N)
            obs: observed (N)
        output:
            RMSE
        """

    # Calculate RMSE
    rmse = np.sqrt(((sim - obs) ** 2).mean())
    return rmse


def BIAS(sim, obs):
    '''
    Bias (mean)
        input:
            sim: simulated (N)
            obs: observed (N)
        output:
            bias
    '''
    b = np.mean(sim - obs)
    return b
