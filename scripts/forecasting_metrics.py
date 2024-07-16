import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_log_error
import scipy.stats as stats

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """Simple error"""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _relative_error(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
):
    """Relative Absolute Error"""
    abs_err = np.abs(_error(actual, predicted))
    abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err_bench + EPSILON)


def _bounded_relative_error(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
):
    """Bounded Relative Absolute Error"""
    abs_err = np.abs(_error(actual, predicted))
    abs_err_bench = np.abs(_error(actual, benchmark))
    return abs_err / (abs_err + abs_err_bench + EPSILON)


def mse(actual: np.ndarray, predicted: np.ndarray):
    """Mean Squared Error"""
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """Root Mean Squared Error"""
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """Mean Absolute Error"""
    return np.mean(np.abs(_error(actual, predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """Normalized Root Mean Squared Error"""
    return rmse(actual, predicted) / (actual.max() - actual.min())


def rmsle(y_true: np.array, y_pred: np.array) -> np.float64:
    """
    The Root Mean Squared Log Error (RMSLE) metric

    :param y_true: The ground truth labels given in the dataset
    :param y_pred: Our predictions
    :return: The RMSLE score
    """
    return mean_squared_log_error(y_true, y_pred, squared=False)


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rrse(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """Root Relative Squared Error"""
    return np.sqrt(
        np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - benchmark))
    )


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """Mean Relative Absolute Error"""
    return np.mean(
        _relative_error(actual=actual, predicted=predicted, benchmark=benchmark)
    )


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """Mean Bounded Relative Absolute Error"""
    return np.mean(
        _bounded_relative_error(actual=actual, predicted=predicted, benchmark=benchmark)
    )


def mda(actual, predicted):
    """
    Calculates the Mean Directional Accuracy (MDA) for two time series.

    Parameters:
    actual (array-like): The actual values for the time series.
    predicted (array-like): The predicted values for the time series.

    Returns:
    float: The MDA value.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    # calculate the signs of the differences between consecutive values
    actual_diff = np.diff(actual)
    actual_signs = np.sign(actual_diff)
    predicted_diff = np.diff(predicted)
    predicted_signs = np.sign(predicted_diff)

    # count the number of times the signs are the same
    num_correct = np.sum(actual_signs == predicted_signs)

    # calculate the MDA value
    mda = num_correct / (len(actual) - 1)

    return mda


def pttest(y, yhat):
    """
    Implementation of the Directional Accuracy Score and Pesaran-Timmermann statistic

    Given arrays with predictions and with true values,
    returns Directional Accuracy Score, Pesaran-Timmermann statistic and its p-value
    """
    size = y.shape[0]
    pyz = np.sum(np.sign(y) == np.sign(yhat)) / size  # Directional Accuracy Score
    py = np.sum(y > 0) / size
    qy = py * (1 - py) / size
    pz = np.sum(yhat > 0) / size
    qz = pz * (1 - pz) / size
    p = py * pz + (1 - py) * (1 - pz)
    v = p * (1 - p) / size
    w = ((2 * py - 1) ** 2) * qz + ((2 * pz - 1) ** 2) * qy + 4 * qy * qz
    pt = (pyz - p) / (np.sqrt(v - w))  # Pesaran-Timmermann statistic
    pval = 1 - stats.norm.cdf(pt, 0, 1)  # Pesaran-Timmermann p-value
    return pyz, pt, pval


def corrcoef(predicted, actual):
    return (
        round(scipy.stats.pearsonr(predicted, actual)[0], 3),  # Pearson's r
        round(scipy.stats.spearmanr(predicted, actual)[0], 3),  # Spearman's rho
        round(scipy.stats.kendalltau(predicted, actual)[0], 3),  # Kendall's tau
    )


def fraction_of_variance_unexplained(actual: np.ndarray, predicted: np.ndarray):
    """
    Fraction of variance unexplained
    """
    return np.var(actual - predicted) / np.var(actual)


def rse(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """Root Relative Squared Error"""
    return np.sqrt(np.sum(np.square(actual - predicted)))


# def relative_corrcoef(predicted, actual, benchmark):
#     return (
#         round(scipy.stats.pearsonr(predicted, actual)[0], 3)/round(scipy.stats.pearsonr(predicted, benchmark)[0], 3),
#         round(scipy.stats.spearmanr(predicted, actual)[0], 3)/round(scipy.stats.spearmanr(predicted, benchmark)[0], 3),
#         round(scipy.stats.kendalltau(predicted, actual)[0], 3)/round(scipy.stats.kendalltau(predicted, benchmark)[0], 3)
#     )


METRICS = {
    "mse": mse,
    "rmse": rmse,
    "nrmse": nrmse,
    "rmspe": rmspe,
    "rmsle": rmsle,
    "rrse": rrse,
    "mae": mae,
    "mape": mape,
    "smape": smape,
    "mrae": mrae,
    "mbrae": mbrae,
    "mda": mda,
    "pttest": pttest,
    "fuv": fraction_of_variance_unexplained,
    "corrcoef": corrcoef,
}


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: np.ndarray = None,
    metrics=METRICS,
):
    results = {}
    for name in metrics:
        try:
            if name in ["mrae", "mbrae", "rrse"]:
                results[name] = METRICS[name](actual, predicted, benchmark)
            elif name == "corrcoef":
                results["pearsonr"] = METRICS[name](actual, predicted)[0]
                results["spearmanr"] = METRICS[name](actual, predicted)[1]
                results["kendalltau"] = METRICS[name](actual, predicted)[2]
            else:
                results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            # print("Unable to compute metric {0}: {1}".format(name, err))
    return results
