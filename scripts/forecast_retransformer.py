import pandas as pd
import numpy as np
import dateutil
from pmdarima.utils import diff_inv
from dateutil.relativedelta import relativedelta


class ForecastRetransformer:
    def __init__(self, df_orig: pd.Series, df_diff: pd.Series):
        self.df_orig = (
            df_orig.reset_index()
            .drop_duplicates()
            .set_index("reference_date")
            .squeeze()
        )
        self.df_diff = (
            df_diff.reset_index()
            .drop_duplicates()
            .set_index("reference_date")
            .squeeze("columns")
        )

    def __inv_diff_1_lag(
        self,
        df_orig_column: pd.Series,
        df_diff_column: pd.Series,
        lagNum: int,
    ) -> pd.Series:
        """
        Method for the absolute change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        lagNum : an integer > 0 indicating which lag to use.


        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """

        rows_beyond = pd.Series(
            index=[
                df_orig_column.index.max() + relativedelta(months=i + 1)
                for i in range(lagNum)
            ]
        )
        df_orig = pd.concat([df_orig_column, rows_beyond])

        return (
            df_orig.shift(periods=lagNum).loc[df_diff_column.index.min() :]
            + df_diff_column.sort_index()
        )

    # def __inv_diff_2_lag(
    #     self,
    #     df_orig_column: pd.Series,
    #     df_diff_column: pd.Series,
    #     lagNum: int,
    # ):
    #     """
    #     Method for the 2nd order change inversion

    #     Parameters
    #     ----------
    #     df_orig_column: original series indexed in time order
    #     df_diff_column: differenced series indexed in time order

    #     Returns
    #     -------
    #     pd.Series: the result of the inverse of the difference arrays.

    #     """
    #     _df_diff_column = df_orig_column.diff(periods=lagNum).dropna()
    #     _inv_diff_series = self.__inv_diff_1_lag(
    #         df_orig_column=_df_diff_column,
    #         df_diff_column=df_diff_column,
    #         lagNum=lagNum,
    #     )

    #     inv_diff_series = self.__inv_diff_1_lag(
    #         df_orig_column=df_orig_column,
    #         df_diff_column=_inv_diff_series,
    #         lagNum=lagNum,
    #     )

    #     return inv_diff_series

    def __inv_perc_diff_1_lag(
        self,
        df_orig_column: pd.Series,
        df_diff_column: pd.Series,
        lagNum: int,
    ) -> pd.Series:
        """
        Method for the absolute change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        lagNum : an integer > 0 indicating which lag to use.


        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """
        rows_beyond = pd.Series(
            index=[
                df_orig_column.index.max() + relativedelta(months=i + 1)
                for i in range(lagNum)
            ]
        )
        df_orig = pd.concat([df_orig_column, rows_beyond])

        return df_orig.shift(periods=lagNum).loc[df_diff_column.index.min() :] * (
            df_diff_column.sort_index() / 100 + 1
        )

    def make_inv_diff(
        self,
        order: int,
        lagNum: int,
    ):
        """
        Method for the 1st and 2nd order change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        order: an integer > 0 indicating the difference order

        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """
        if order == 1:
            return self.__inv_diff_1_lag(
                df_orig_column=self.df_orig, df_diff_column=self.df_diff, lagNum=lagNum
            )
        # elif order == 2:
        #     return self.__inv_diff_2_lag(
        #         df_orig_column=self.df_orig,
        #         df_diff_column=self.df_diff,
        #         lagNum=lagNum
        #     )
        else:
            return pd.Series()

    def make_inv_perc_diff(self, lagNum: int, order: int):
        """
        Method for the 1st and 2nd order change inversion

        Parameters
        ----------
        df_orig_column: original series indexed in time order
        df_diff_column: differenced series indexed in time order
        order: an integer > 0 indicating the difference order

        Returns
        -------
        pd.Series: the result of the inverse of the difference arrays.

        """
        if order == 1:
            return self.__inv_perc_diff_1_lag(
                df_orig_column=self.df_orig, df_diff_column=self.df_diff, lagNum=lagNum
            )
        else:
            return pd.Series()
