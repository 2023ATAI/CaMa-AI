import numpy as np
import xarray as xr
import torch
from scipy.stats import ttest_rel



def correlation(o, s):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    if isinstance(s, np.ndarray):
        s = xr.DataArray(s)
    if isinstance(o, np.ndarray):
        o = xr.DataArray(o)
    corr = xr.corr(s, o)

    return corr
def correlation_R2(o,s):
    """
    correlation coefficient R2
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    if isinstance(s, np.ndarray):
        s = xr.DataArray(s)
    if isinstance(o, np.ndarray):
        o = xr.DataArray(o)

    return xr.corr(s,o)**2
def unbiased_rmse(y_true, y_pred):
    predmean = np.nanmean(y_pred)
    targetmean = np.nanmean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.nanmean((predanom-targetanom)**2))



def _rmse(y_true,y_pred):
    predanom = y_pred
    targetanom = y_true
    return np.sqrt(np.nanmean((predanom-targetanom)**2))

def _bias(y_true,y_pred):
    bias = np.nanmean(np.abs(y_pred-y_true))
    return bias

def GetKGE(y_test, y_pred):
    y_test_mean = np.mean(y_test)
    y_pred_mean = np.mean(y_pred)
    y_test_std = np.std(y_test)
    y_pred_std = np.std(y_pred)

    r = np.corrcoef(y_test, y_pred)[0, 1]

    kge_value = 1 - np.sqrt((r - 1) ** 2 +
                            ((y_pred_std / y_test_std) - 1) ** 2 +
                            ((y_pred_mean / y_test_mean) - 1) ** 2)
    return kge_value

def KGESS(y_true, y_pred):
    KGE_benchmark = -0.41
    KGE_model = GetKGE(y_true, y_pred)
    if np.isnan(KGE_model):
        return np.nan
    return (KGE_model - KGE_benchmark) / (1 - KGE_benchmark)

def GetMAE(y_true,y_pred):
    """
    Calculate the Mean Absolute Error (MAE).

    Parameters:
        y_true (list or array): Actual observed values (ground truth / true targets)
        y_pred (list or array): Predicted values from the model

    Returns:
        float: Mean Absolute Error (MAE)
    """
    n = len(y_true)
    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
    return mae

def GetNSE(observed,simulated):

    if isinstance(observed, torch.Tensor):
        observed = observed.cpu()
        simulated = simulated.cpu()
        observed = observed.numpy()
        simulated = simulated.numpy()
    mask = observed != 0
    observed = observed[mask]
    simulated = simulated[mask]
    mean_observed = np.mean(observed)
    numerator = np.sum((simulated - observed)**2)
    denominator = np.sum((observed - mean_observed)**2)
    nse = 1 - numerator / denominator
    return nse

def GetP(y_test,y_pred):
    # Assume that baseline_pred is the prediction from the baseline model
    baseline_pred = [y_test.mean()] * len(y_test)  # The baseline model predicts the mean value of y_test

    errors_model = y_test - y_pred
    errors_baseline = y_test - baseline_pred

    t_statistic, p_value = ttest_rel(errors_model, errors_baseline)

    return p_value
