import os
import re
import pandas as pd
import numpy as np

import json

from typing import List

from dateutil.relativedelta import relativedelta
from datetime import datetime

from sklearn import preprocessing

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import LinearRegression
from lineartree import LinearForestRegressor
from lineartree import LinearBoostRegressor
from lineartree import LinearTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

from forecast_retransformer import ForecastRetransformer
from forecasting_metrics import evaluate
from pathlib import Path

pd.options.mode.chained_assignment = None
import subprocess
import sys
import argparse


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def convert_to_datetime(df: pd.DataFrame, colnames: List[str]) -> pd.DataFrame:
    for col in colnames:
        df[col] = pd.to_datetime(df[col])
    return df


def __arima_feed(series, h=6):
    series = series.dropna()
    arima_model = pm.auto_arima(series, stepwise=True)
    forecast = arima_model.predict(n_periods=h)
    forecast_index = pd.date_range(
        series.index[-1] + relativedelta(months=1), periods=h, freq="MS"
    )
    forecast_series = pd.Series(forecast, index=forecast_index)
    # series = series.append(forecast_series)
    # return series
    return forecast_series


# def load_state(log_path: str):
#     try:
#         with open(log_path, "r") as logfile:
#             log = json.load(logfile)
#     except (json.JSONDecodeError, FileNotFoundError):
#         log = []
#     return log


# def save_state(state_list, log_path):
#     log = load_state(log_path)
#     log_update = log + state_list
#     with open(log_path, "w") as logfile:
#         json.dump(log_update, logfile)


def main_body(df_long):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--log-path")
    # args = parser.parse_args()

    # log_listdir = os.listdir(args.log_path)
    # log = []
    # for log_path in log_listdir:
    #     log = log + load_state(os.path.join(args.log_path, log_path))

    # log_df = pd.DataFrame(
    #     log,
    #     columns=["y_code", "reference_date", "model", "x", "start_index", "start_date"],
    # )

    ds_fname = set(df_long["ds_fname"]).pop()
    ds_long = df_long.drop(columns=["ds_fname"])

    model_output_fname = (
        os.path.basename(ds_fname)
        .replace("csv", "json")
    )

    if model_output_fname in os.listdir(model_dirpath):
        print(f"WARNING: {model_output_fname} already exists.")
        return

    y_fields = set(ds_long.loc[ds_long["IsTarget"] == 1]["series_code"])
    assert len(y_fields) == 1

    y_code = y_fields.pop()
    reference_date = ds_long.loc[ds_long["series_code"] == y_code][
        "reference_date"
    ].max() + relativedelta(months=1)
    # print(f'Y field: {y_code}, reference date: {reference_date.strftime("%Y-%m-%d")}')

    # find file with raw series for retransformation
    y_id = y_code.split("_")[0]

    fs_method = (
        re.search(f'{reference_date.strftime("%Y%m%d")}_(.*)', ds_fname)
        .group(1)
        .split("_")[0]
    )

    match_substr = re.search(f"real_time_vintage_(.*)_{fs_method}_", ds_fname).group(1)
    find_substr = f'df_clean_actual_real_time_vintage_{match_substr.split("_")[0]}_{match_substr.split("_")[-1]}'
    matching_src_fpaths = [s for s in src_listdir if find_substr in s]
    assert len(matching_src_fpaths) == 1

    src_df = pd.read_parquet(matching_src_fpaths[0])
    src_df = convert_to_datetime(src_df, ["reference_date"])

    reference_date = ds_long.loc[ds_long["series_code"] == y_code][
        "reference_date"
    ].max()
    fcst_ref_dates = [
        (reference_date - relativedelta(months=i)).strftime("%Y-%m-%d")
        for i in range(n_periods - 1, -1, -1)
    ]
    # print(f'INFO: Test reference dates: {fcst_ref_dates}')

    df = ds_long.drop(columns=["IsTarget"]).rename(columns={"value": "variable_value"})

    df_pivot = df[["reference_date", "series_code", "variable_value"]].pivot(
        index="reference_date", columns="series_code", values="variable_value"
    )
    df_pivot = (
        convert_to_datetime(df_pivot.reset_index(), ["reference_date"])
        .set_index("reference_date")
        .sort_index()
    )

    # AR model training
    # print(f"INFO: Estimating ARIMA...")

    benchmark_df = pd.DataFrame()
    df_train = df_pivot  # .drop(reference_date)
    for ref_date in fcst_ref_dates:  # [:-1]:
        # split between train and test subsets
        y_train, y_test = (
            df_train[[y_code]].loc[df_train.index < ref_date],
            df_train[[y_code]].loc[df_train.index >= ref_date],
        )

        # forecast inference
        fcst_df = y_train.apply(__arima_feed, h=y_test.shape[0], axis=0)
        fc_series = fcst_df.squeeze(axis=1)

        benchmark_df = pd.concat([benchmark_df, fc_series], axis=1)

    # print("INFO: Completed.")

    cols = ["model" + str(i + 1) for i in range(benchmark_df.shape[1])]
    benchmark_df.columns = cols

    benchmark_backcast = pd.Series(
        np.diag(benchmark_df), index=benchmark_df.index, dtype=float
    )
    benchmark_backcast.index.name = "reference_date"

    # iterating over features
    features = set(df_pivot.drop(columns=[y_code]).columns)
    # state_list = []
    exp_i, exp_results = 0, []
    for x in features:
        time_series = df_pivot[x]
        # Determine the length of the time series
        series_length = df_pivot.loc[: fcst_ref_dates[0]].shape[0]
        starting_points = np.random.randint(
            window_size + 1, series_length - window_size, num_starting_points
        )

        for start_index in starting_points:
            # Replace the values in the random window with the mean of values preceding that period
            mean_value = time_series.iloc[:start_index].mean()

            df_pivot_pf = df_pivot.copy()
            df_pivot_pf[x].iloc[start_index : start_index + window_size] = mean_value

            # iterating over different algorithms
            for model in estimators:
                exp_i = exp_i + 1
                # check = (
                #     log_df.loc[
                #         (log_df["y_code"] == y_code)
                #         & (
                #             log_df["reference_date"]
                #             == reference_date.strftime("%Y%m%d")
                #         )
                #         & (log_df["model"] == type(model).__name__)
                #         & (log_df["x"] == x)
                #     ].shape[0]
                #     >= num_starting_points
                # )

                # if check:
                #     print(
                #         "INFO: Pixel flipping done for current setup. Moving to next iteration."
                #     )
                #     continue

                forecasts_df, coef_df = pd.DataFrame(), pd.DataFrame()

                start_date = (
                    df_pivot_pf.reset_index()
                    .iloc[start_index]["reference_date"]
                    .strftime("%Y%m%d")
                )
                # model_output_fname = (
                #     os.path.basename(ds_fname)
                #     .replace("df_", f'fcst_{type(model).__name__}_{reference_date.strftime("%Y%m%d")}_', 1)
                #     .replace(".csv", "")
                #     + f'_feature_{x}_start_index_{start_index}_{start_date}.json'
                # )

                # if model_output_fname in os.listdir(model_dirpath):
                #     print(f"WARNING: {model_output_fname} already exists.")
                #     continue

                # print(f"INFO: Estimating {type(model).__name__}...")

                assert_index = (
                    df_pivot_pf[x].iloc[start_index : start_index + window_size].index
                )

                # cascading model training

                for ref_date in fcst_ref_dates:
                    # split between train and test subsets
                    df_train, df_test = (
                        df_pivot_pf.loc[df_pivot_pf.index < ref_date],
                        df_pivot_pf.loc[df_pivot_pf.index >= ref_date],
                    )

                    df_train.index = pd.DatetimeIndex(df_train.index.values, freq="MS")
                    df_test.index = pd.DatetimeIndex(df_test.index.values, freq="MS")

                    X_train, y_train = (
                        df_train.drop(columns=[y_code]),
                        df_train[[y_code]],
                    )
                    X_test, y_test = (
                        df_test.drop(columns=[y_code]),
                        df_test[[y_code]],
                    )

                    # # feature scaling
                    # scaler = preprocessing.RobustScaler()
                    # X_train_ss = scaler.fit_transform(X_train.values)

                    # scaler = preprocessing.RobustScaler()
                    # X_test_ss = scaler.fit_transform(X_test.values)

                    # model fitting
                    try:
                        model.fit(X_train, y_train.values.ravel())
                    except np.linalg.LinAlgError:
                        continue

                    # forecast inference
                    fcst = model.predict(X_test)
                    fc_series = pd.Series(fcst, index=y_test.index)
                    forecasts_df = pd.concat([forecasts_df, fc_series], axis=1)

                    # coefficients extraction
                    try:
                        coef_series = pd.Series(
                            np.round(model.coef_, 5), index=X_train.columns
                        )
                    except AttributeError:
                        coef_series = pd.Series(
                            np.round(model.feature_importances_, 5),
                            index=X_train.columns,
                        )
                    except ValueError:
                        coef_series = pd.Series(
                            np.round(
                                model.base_estimator_.fit(X_train, y_train).coef_[0],
                                5,
                            ),
                            index=X_train.columns,
                        )
                    coef_df = pd.concat([coef_df, coef_series], axis=1)

                cols = ["model" + str(i + 1) for i in range(forecasts_df.shape[1])]
                forecasts_df.columns = cols
                coef_df.columns = cols

                if len(np.diag(forecasts_df)) != n_periods:
                    continue

                # forecasts retransformation;
                forecast_df_diag = pd.Series(
                    np.diag(forecasts_df), index=forecasts_df.index, dtype=float
                )
                forecast_df_diag.index.name = "reference_date"

                y_df_orig = (
                    src_df.loc[src_df["variable_id"] == y_id][
                        ["reference_date", "variable_id", "variable_value"]
                    ]
                    .pivot(
                        index="reference_date",
                        columns="variable_id",
                        values="variable_value",
                    )
                    .loc[df_train.index.min() : forecast_df_diag.index.max()]
                    .sort_index()
                )
                assert y_df_orig.shape[0] > 0
                y_df_orig.index.name = "reference_date"

                b_fr = ForecastRetransformer(
                    df_orig=y_df_orig, df_diff=forecast_df_diag
                )
                bench_fr = ForecastRetransformer(
                    df_orig=y_df_orig, df_diff=benchmark_backcast
                )

                re_match = re.search("lag_(\d+)", y_code)
                if re_match:
                    offset = int(re_match.group(1))

                if list(filter(re.compile(f".*perc_diff_lag").match, [y_code])):
                    retransf_fcst = b_fr.make_inv_perc_diff(order=1, lagNum=offset)
                    retransf_bench_bcst = bench_fr.make_inv_perc_diff(
                        order=1, lagNum=offset
                    )
                elif list(filter(re.compile(f".*diff_lag").match, [y_code])):
                    retransf_fcst = b_fr.make_inv_diff(order=1, lagNum=offset)
                    retransf_bench_bcst = bench_fr.make_inv_diff(order=1, lagNum=offset)
                elif y_code == y_id:
                    retransf_fcst = forecast_df_diag
                    retransf_bench_bcst = benchmark_backcast
                else:
                    print("WARNING: The transformation was not recognized")
                    break

                retransf_fcst = retransf_fcst.loc[forecast_df_diag.index]
                retransf_bench_bcst = retransf_bench_bcst.loc[benchmark_backcast.index]

                df_pivot.index = pd.to_datetime(df_pivot.index)
                y_df_orig.index = pd.to_datetime(y_df_orig.index)
                actual = df_pivot[y_code].loc[forecast_df_diag.index]
                retransf_actual = y_df_orig[y_id].loc[forecast_df_diag.index]

                eval_out = evaluate(
                    actual=retransf_actual,
                    predicted=retransf_fcst,
                    benchmark=retransf_bench_bcst,
                    metrics=[
                        "rmse",
                        "rmspe",
                        "mape",
                        "smape",
                        "mrae",
                        "rrse",
                        "corrcoef",
                    ],
                )
                assert sum(np.isnan(list(eval_out.values()))) == 0

                accuracy_report = {
                    "y_id": y_id,
                    "y_field": y_code,
                    "target_reference_date": reference_date.strftime("%Y-%m-%d"),
                    "estimator": type(model).__name__,
                    "fs_method": fs_method,
                    "coefficients": coef_df[f"model{n_periods}"].to_dict(),
                    "eval_metrics": eval_out,
                    "x": x,
                    "start_index": int(start_index),
                    "start_date": start_date,
                }

                y_pred = forecast_df_diag
                y_pred_retr = retransf_fcst
                y = df_pivot[y_code]
                y_src = y_df_orig[y_id]

                y_pred.index = pd.to_datetime(y_pred.index).strftime("%Y-%m-%d")
                y_pred_retr.index = pd.to_datetime(y_pred_retr.index).strftime(
                    "%Y-%m-%d"
                )
                y.index = pd.to_datetime(y.index).strftime("%Y-%m-%d")
                y_src.index = pd.to_datetime(y_src.index).strftime("%Y-%m-%d")

                accuracy_report.update(
                    {
                        "forecast": forecast_df_diag.to_dict(),
                        "retransf_forecast": retransf_fcst.to_dict(),
                        "y_actual": y.to_dict(),
                        "y_src": y_src.to_dict(),
                        "ds_fname": ds_fname,
                        # "out_fname": model_output_fname,
                    }
                )

                # with open(os.path.join(model_dirpath, model_output_fname), "w") as f:
                #     json.dump(accuracy_report, f)
                # state_list.append(
                #     {
                #         "y_code": y_code,
                #         "reference_date": reference_date.strftime("%Y%m%d"),
                #         "model": type(model).__name__,
                #         "x": x,
                #         "start_index": int(start_index),
                #         "start_date": start_date,
                #     }
                # )
                exp_results.append({f"experiment_{exp_i}": accuracy_report})

    # save_state(
    #     state_list=state_list,
    #     log_path=os.path.join(
    #         args.log_path, f'/log_{os.path.basename(ds_fname).split(".")[0]}.json'
    #     ),
    # )
    with open(os.path.join(model_dirpath, model_output_fname), "w") as f:
        json.dump(exp_results, f)
    print(17 * "-")


from multiprocessing import Pool

if __name__ == "__main__":

    # path = Path(os.path.dirname(__file__))
    # path = "/home2/faculty/ejowik/mts-nowcasting/"
    # path = "/Users/ejowik001/Desktop/pf-comp/"
    path = "/root/pf-comp/"

    src_path = os.path.join(path, "data/03_intermediate/data_source/")
    ds_path = os.path.join(path, f"data/4_features/")
    model_dirpath = os.path.join(path, f"data/7_pixel_flipping/")

    n_periods = 12
    window_size = 12
    num_starting_points = 1 

    estimators = [
        LinearRegression(),
        ElasticNet(),
        LinearTreeRegressor(base_estimator=LinearRegression()),
        LinearForestRegressor(base_estimator=LinearRegression(), max_features=None),
        LinearBoostRegressor(base_estimator=LinearRegression()),
        RandomForestRegressor(),
        XGBRegressor(),
    ]

    src_listdir = []
    for path, subdirs, files in os.walk(src_path):
        for name in files:
            if not name.endswith(".parquet"):
                continue
            src_listdir.append(os.path.join(path, name))


    ds_listdir = []
    for path, subdirs, files in os.walk(ds_path):
        for name in files:
            if not name.endswith(".csv"):
                continue
            if path.split("/")[-1].startswith("baseline"):
                continue
            ds_listdir.append(os.path.join(path, name))

    ds_listdir = [i for i in ds_listdir if os.path.basename(i) != "MRMR_nfeatures_auto.csv"]


    list_df = []
    for ds_fname in ds_listdir:
        if not ds_fname.endswith(".csv"):
            continue

        # print(ds_fname)
        ds_long = pd.read_csv(ds_fname)
        if ds_long.shape[1] == 1:
            ds_long = pd.read_csv(ds_fname, sep=";")
            ds_long["reference_date"] = ds_long["reference_date"].str.split(
                " ", expand=True
            )[0]

        try:
            ds_long = convert_to_datetime(ds_long, ["reference_date"])
        except:
            ds_long = ds_long.loc[
                : (ds_long.loc[ds_long["reference_date"] == "reference_date"].index[0] - 1)
            ]
            ds_long = convert_to_datetime(ds_long, ["reference_date"])
            ds_long["IsTarget"] = ds_long["IsTarget"].astype(int)
            ds_long["value"] = ds_long["value"].astype(float)

        ds_long["ds_fname"] = ds_fname

        list_df.append(ds_long)



    with Pool(4) as p:
        # results = p.map(main, list_df)
        p.map(main_body, list_df)

        # for result in results:
        #     print(results)