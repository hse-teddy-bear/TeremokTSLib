"""
Model
---------
A modular object which consists of multiple layers of Prophet and CatBoost models and coefficients for converting forcasted consumption to order.
Structure is STR decomposition of time series: adding layers of models one by one on base layer (ewma).
"""


# -----------------------------------------------------------------------
# dependencies


import pandas as pd
import numpy as np
import pickle
import optuna
import itertools
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas.api.types as ptypes
import statsmodels.api as sm
from catboost import CatBoostRegressor
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.deterministic import CalendarFourier


# -----------------------------------------------------------------------
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

def _calc_rolling_ewma(ts: pd.Series, window_size=60) -> pd.Series:
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

def _train_prophet_model(
        data: pd.DataFrame, 
        date_col: str, 
        target_col: str, 
        trend_col: str, 
        holidays: pd.DataFrame,
        is_tuning: bool=False,
    ) -> Prophet:
    """
    Returns a trained Prophet model on given time series data.
    """

    # Supress LOGGING
    logging.getLogger("cmdstanpy").disabled = True

    # Setting threshholds for time series split (train set)
    unique_dates = sorted(data[date_col].unique())
    train_threshold =  unique_dates[int(len(unique_dates) * 0.85)] #85%
    train_data = data.loc[data[date_col] <= train_threshold]

    #detrending data
    train_data[target_col] = train_data[target_col] - train_data[trend_col]
    train_data = train_data[[date_col, target_col]]
    train_data.columns = ['ds', 'y']

    # Hyperparams tuning for Prophet
    if is_tuning:
        param_grid = {  
            'changepoint_prior_scale': [0.01, 0.025, 0.03, 0.04, 0.05],
            'seasonality_prior_scale': [0.5, 1, 2, 3, 5, 8, 10],
            'holidays_prior_scale': [5, 10, 30, 50],
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params, 
                        growth='flat',
                        holidays=holidays, 
                        daily_seasonality=False,
                        yearly_seasonality=False)
            m.add_seasonality(name='weekly', period=7, fourier_order=4)
            m.add_seasonality(name='monthly', period=30, fourier_order=4)
            # m.add_seasonality(name='yearly', period=365, fourier_order=4) <-- yearly trend is deprecated, as short-term trend captures it
            m.fit(train_data)

            y_pred = m.predict(train_data[['ds']])['yhat']
            y_real = train_data['y']
            rmse_metric = mean_squared_error(y_real, y_pred, squared=False)
            rmses.append(rmse_metric)
        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        # Initialize the Prophet model
        best_cps = tuning_results.sort_values(by='rmse').iloc[0]['changepoint_prior_scale']
        best_sps = tuning_results.sort_values(by='rmse').iloc[0]['seasonality_prior_scale']
        best_hps = tuning_results.sort_values(by='rmse').iloc[0]['holidays_prior_scale']

    else:
        best_cps = 0.025
        best_sps = 5
        best_hps = 10

    model = Prophet(seasonality_mode='additive',
                    growth='flat',
                    changepoint_prior_scale=best_cps, 
                    seasonality_prior_scale=best_sps, 
                    holidays_prior_scale=best_hps,
                    daily_seasonality=False,
                    yearly_seasonality=False,
                    holidays=holidays)

    # Add seasonality
    model.add_seasonality(name='weekly', period=7, fourier_order=4)
    model.add_seasonality(name='monthly', period=30, fourier_order=4)

    # Fit the model
    model.fit(train_data)
    return model

def _show_pacf(time_series: list, 
              nlags: int=30, 
              alpha: float=0.1
    ) -> None:
    """
    Shows PACF function on lags of second residuals (for CatBoost layer).
    If values of autocorrelation function are closer to 1, than past data heavily influences the future.
    If so, CatBoost model will dramatically increase accuracy of the ensemble.
    """
    curr_fig, curr_ax = plt.subplots(figsize=(15, 5))
    sm.graphics.tsa.plot_pacf(time_series, lags=nlags, ax=curr_ax, alpha=alpha)
    plt.xticks(range(0, nlags+1))
    plt.xlabel('lag number')
    plt.ylabel('value')
    plt.title(f'PACF on Residuals, lags={nlags}')
    plt.grid(True)
    plt.show()


# -----------------------------------------------------------------------
# Model class


class Model:

    def __init__(
            self, 
    )-> None:
        self.prophet_model = Prophet(),
        self.catboost_model = CatBoostRegressor(),
        self.beta_coefficient = float(),
        self.inference_cols = list(),
        self.is_trained = False,
        self.train_date = str(),

    def __str__(self) -> str:
        return '<TeremokTSLib Model>'

    def __repr__(self) -> str:
        return '<TeremokTSLib Model>'
    
    def train(
            self,
            data: pd.DataFrame,
            catboost_params: list,
            prophet_params: list,
            ewma_length: int=10,
            interpolate_spaces: bool=True,
            holidays_df: pd.DataFrame=make_holidays_df(year_list=[1990 + i for i in range(40)], country='RU'),
            show_residuals_pacf: int=0,
            save_model: bool=False,
    ) -> None:
        """
        Method for training a TeremokTSLib models inside mother Model on given time series data.
        Note: you should pass data in correct format (See docs).

        Examples
        --------
        >>> import TeremokTSLib as tts
        >>> model = tts.Model()
        >>> model.train(data=my_df)
        """


        # checking whether input params are in correct format
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas DataFrame. Provided type: {type(data)}")
        
        if not isinstance(catboost_params, list):
            raise ValueError(f"catboost_params must be a list. Provided type: {type(catboost_params)}")
        
        if not isinstance(prophet_params, list):
            raise ValueError(f"prophet_params must be a list. Provided type: {type(prophet_params)}")
        
        if not isinstance(ewma_length, int):
            raise ValueError(f"ewma_length must be an integer. Provided type: {type(ewma_length)}")
        
        if not isinstance(interpolate_spaces, bool):
            raise ValueError(f"interpolate_spaces must be a boolean. Provided type: {type(interpolate_spaces)}")
        
        if not isinstance(show_residuals_pacf, int):
            raise ValueError(f"show_residuals_pacf must be a integer. Provided type: {type(show_residuals_pacf)}")
        
        if not isinstance(save_model, bool):
            raise ValueError(f"save_model must be a boolean. Provided type: {type(save_model)}")

        if ewma_length <= 0:
            raise ValueError(f"ewma_length must be a positive integer. Provided value: {ewma_length}")
        
        if show_residuals_pacf < 0:
            raise ValueError(f"show_residuals_pacf must be a positive integer or zero. Provided value: {ewma_length}")


        # checking whether input dataframe is in correct format
        if data.shape[0] < ewma_length:
            raise ValueError(f"Input data must have at least {ewma_length} rows")

        if len(data.columns) != 2:
            raise ValueError("Input data must have 2 columns: 'date' and 'consumption'")
        
        if 'date' not in data.columns:
            raise ValueError("Input data must contain 'date' column")
        
        if 'consumption' not in data.columns:
            raise ValueError("Input data must contain 'consumption' column")
        

        # checking input daily data for sortedness
        data = data.sort_values(by='date').reset_index(drop=True)


        # checking input daily time series data for missing dates (spaces)
        if _check_spaces(data) != None:
            is_continuous = False
            print('Input data has missing dates. Proceeding to interpolation.')
        elif data.isnull().values.any():
            is_continuous = False
            print('Input data has NaNs. Proceeding to interpolation.')            
        else:
            is_continuous = True
            print('Input data is continuous. Proceeding to training.')

        if interpolate_spaces == True and is_continuous == False:
            data = _add_missing_dates(data)
            data = _interpolate_spaces(data)
        elif interpolate_spaces == False and is_continuous == False:
            raise Warning("Interpolation was switched off! Data should be continuous for correct training process.")


        # calculating ewma (base layer) as Trend layer
        data['trend'] = _calc_rolling_ewma(data['consumption'], window_size=ewma_length)
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)


        # getting first residual and training Prophet model (first layer) as Seasonality layer
        fb_model = _train_prophet_model(data=data,
                                        date_col='date',
                                        target_col='consumption',
                                        trend_col='trend',
                                        holidays=holidays_df,
                                        is_tuning=True)
        self.prophet_model = fb_model


        # getting second residual
        ts_data_prophet = data[['date']].rename(columns={'date':'ds'})
        prophet_predictions = self.prophet_model.predict(ts_data_prophet[['ds']])
        data['seasonality'] = prophet_predictions['additive_terms']
        data['residual'] = data['consumption'] - data['trend'] - data['seasonality']
        ts_for_training = data[['date', 'residual']]
        if show_residuals_pacf != 0:
            _show_pacf(ts_for_training.residual[::-1].values.tolist(), nlags=show_residuals_pacf, alpha=0.05)

        
        # preparing training data for CatBoost


        # training CatBoost model (second layer) as Residual layer



        # beta coefficient optimization

        self.is_trained = True
        self.train_date = '00-00-0000'

    def itertest(
            self,
            data: pd.DataFrame,
            days_till_perish: int=2,
            alpha_coefficient: int=4,
    ) -> float:
        """
        Returns result of stress-testing trained model in format of weighted loss metric.
        Formula: loss_metric = write_off_count + sale_stop_count * alpha_coefficient.
        Increase alpha_coefficient parameter if you want penalty for sale_stops to increase.
        
        By default alpha_coefficient is 4.
        """
        pass

    def predict_consumption(
            self,
            data: pd.DataFrame,
    ) -> list:
        """
        Returns predicted consumption on day T+1 (prediction day)
        """

        # checking if all input columns are present
        # getting consumption prediction 
        pass

    def predict_order(
            self,
            data: pd.DataFrame,
            k_coef: int=1,
    ) -> list:
        """
        Returns recommended orders on day T+1 (prediction day) based on forcasted consumption on day T+1.
        Note: input data should contain all necessary columns and stock_left column. 
        To view them, use view_inference_cols() method.

        Examples
        --------
        >>> model = tts.Model()
        >>> model.train(data=train_data)
        >>> order = model.predict_order(data=inference_data, k_coef=10)
        """

        # checking if all input columns are present
        # getting consumption prediction 
        # converting predicted consumption to order in boxes with k and beta coefficients
        pass

    def view_inference_cols(
            self,
    ) -> list:
        """
        Returns a list of necessary columns for inference.
        """
        return self.inference_cols