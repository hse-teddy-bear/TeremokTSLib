"""
Support
---------
Support functions for ensemble model class.
"""

import pandas as pd
import numpy as np


# ----------------------------------------------------
# support methods

def _check_spaces(data: pd.DataFrame) -> pd.Timestamp:
    """
    Returns timestamp of first occured missing date. If no missing dates, returns None.
    """
    data = data.sort_values(by='date')
    unique_dates = data['date'].dt.date.unique()
    prev_date = unique_dates[0]
    for curr_date in unique_dates[1:]:
        expected_date = prev_date + pd.Timedelta(days=1)
        if curr_date != expected_date:
            missing_date = expected_date
            return missing_date
        prev_date = curr_date

def _add_missing_dates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe with added missing dates. Consumption values are NaN.
    """
    data = data.sort_values(by='date').reset_index(drop=True)
    full_date_range = pd.date_range(start=data['date'].min(), end=data['date'].max())
    data = data.set_index('date')
    data = data.reindex(full_date_range).reset_index()
    data = data.rename(columns={'index': 'date'})
    return data
    
def _interpolate_spaces(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns data with interpolated missing days. Means of 5 prev and 5 next daily values.
    """
    data = data.sort_values(by='date').reset_index(drop=True)
    for idx, row in data.iterrows():
        if pd.isna(row['consumption']):
            current_date = row['date']
            prev_5_days = data[(data['date'] < current_date) & (data['consumption'].notna())].tail(5)
            next_5_days = data[(data['date'] > current_date) & (data['consumption'].notna())].head(5)
            relevant_days = pd.concat([prev_5_days, next_5_days])
            mean_value = relevant_days['consumption'].mean()
            data.at[idx, 'consumption'] = mean_value
    return data

def _calc_rolling_ewma(ts: pd.Series, window_size: int=60) -> pd.Series:
    """
    Calculate EWMA for each rolling window.
    """
    ewma_rolling = pd.Series()
    for i in range(window_size):
        ewma_rolling = pd.concat([ewma_rolling, pd.Series(np.nan, index=[i])])
    for i in range(len(ts) - window_size):
        window = ts.iloc[i:i + window_size]
        ewma_value = window.ewm(span=window_size, min_periods=window_size).mean().iloc[-1]
        ewma_rolling = pd.concat([ewma_rolling, pd.Series(ewma_value, index=[i+window_size])])
    return ewma_rolling