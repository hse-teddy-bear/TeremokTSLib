"""
Model
---------
A modular object which consists of multiple layers of Prophet and CatBoost models and coefficients for converting forcasted consumption to order.
Structure is STR decomposition of time series: adding layers of models one by one on base layer (ewma).
"""


# -----------------------------------------------------------------------
# dependencies


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import os
import optuna
import itertools
import logging
import matplotlib.pyplot as plt
import statsmodels.api as sm
from catboost import CatBoostRegressor
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.deterministic import CalendarFourier
from datetime import datetime, date
from joblib import Parallel, delayed

from TeremokTSLib.itertest import optimization


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

def _calculate_lagged_consumption_and_ewma(
        df: pd.DataFrame, 
        n_lags_list: list, 
        ewma_length: int
    ) -> pd.DataFrame:
    """
    Prepares dataset for stresstest. Returns a prepd dataframe.
    """
    # Ensure the dataframe is sorted by date
    df = df.sort_values(by='date').reset_index(drop=True)
    # Calculate EWMA for the current day (T+1)
    df[f'ewma({ewma_length})(T+1)'] = _calc_rolling_ewma(df['consumption'], window_size=ewma_length).shift(-1)
    # Add lagged consumption columns and calculate EWMA for each lag
    for lag in n_lags_list:
        shifted_cons = df['consumption'].shift(lag)
        lag_col = f'cons(T-{lag-1})'
        ewma_col = f'ewma({ewma_length})(T-{lag-1})'
        df[ewma_col] = _calc_rolling_ewma(shifted_cons, window_size=ewma_length).shift(-1)
        df[lag_col] = shifted_cons
    return df

def _train_prophet_model(
        data: pd.DataFrame, 
        date_col: str, 
        target_col: str, 
        trend_col: str, 
        holidays: pd.DataFrame,
        is_tuning: bool=False,
        fb_njobs: int=1,
    ) -> Prophet:
    """
    Returns a trained Prophet model on given time series data.
    """

    # Supress LOGGING
    logging.getLogger("cmdstanpy").disabled = True

    # Setting threshholds for time series split (train set)
    unique_dates = sorted(data[date_col].unique())
    train_threshold =  unique_dates[int(len(unique_dates) * 0.99)] #all input data is train data for prophet
    train_data = data.loc[data[date_col] <= train_threshold]

    #detrending data
    train_data[target_col] = train_data[target_col] - train_data[trend_col]
    train_data = train_data[[date_col, target_col]]
    train_data.columns = ['ds', 'y']

    # Hyperparams tuning for Prophet
    if is_tuning:
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").disabled = True
        param_grid = {  
            'changepoint_prior_scale': [0.01, 0.025, 0.03, 0.05, 0.1],
            'seasonality_prior_scale': [0.5, 1, 2, 3, 5, 8, 10],
            'holidays_prior_scale': [5, 10, 30, 50],
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use gridsearch to evaluate all parameters
        def gridsearch(params):
            m = Prophet(**params, 
                        growth='flat',
                        holidays=holidays, 
                        daily_seasonality=False,
                        yearly_seasonality=False,
                        uncertainty_samples=None,)
            m.add_seasonality(name='weekly', period=7, fourier_order=4)
            m.add_seasonality(name='monthly', period=30, fourier_order=4)
            #m.add_seasonality(name='yearly', period=365, fourier_order=10) <- never include this fucker!
            m.fit(train_data)

            y_pred = m.predict(train_data[['ds']])['yhat']
            y_real = train_data['y']
            rmse_metric = root_mean_squared_error(y_real, y_pred)
            return rmse_metric

        # Parallel training
        rmses = Parallel(n_jobs=fb_njobs)(delayed(gridsearch)(params) for params in all_params)

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses

        # Initialize the Prophet model
        best_cps = tuning_results.sort_values(by='rmse').iloc[0]['changepoint_prior_scale']
        best_sps = tuning_results.sort_values(by='rmse').iloc[0]['seasonality_prior_scale']
        best_hps = tuning_results.sort_values(by='rmse').iloc[0]['holidays_prior_scale']

    else:
        best_cps = 0.5
        best_sps = 5
        best_hps = 10

    model = Prophet(seasonality_mode='additive',
                    growth='flat',
                    changepoint_prior_scale=best_cps, 
                    seasonality_prior_scale=best_sps, 
                    holidays_prior_scale=best_hps,
                    daily_seasonality=False,
                    yearly_seasonality=False,
                    holidays=holidays,
                    uncertainty_samples=None,)

    # Add seasonality
    model.add_seasonality(name='weekly', period=7, fourier_order=4)
    model.add_seasonality(name='monthly', period=30, fourier_order=4)
    #model.add_seasonality(name='yearly', period=365, fourier_order=10) <- never include this fucker!

    # Fit the model
    model.fit(train_data)
    return model

def _show_pacf(
        time_series: list, 
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

def _create_daily_lags(
        df: pd.DataFrame, 
        lag_columns_input: list, 
        n_days: int=1, 
        n_days_list: list=[], 
        is_backwards: bool=True, 
        continuous: bool=True,
    ) -> pd.DataFrame:
    """
    Returns transformed dataframe with lag values of target.
    """
    if continuous:
        for lag_column in lag_columns_input:
            for i in range(1, n_days + 1):
                if is_backwards == True:
                    df[f'{lag_column}(T-{i})'] = df[lag_column].shift(periods=i)
                elif is_backwards == False:
                    df[f'{lag_column}(T+{i})'] = df[lag_column].shift(periods=-i)
    elif continuous == False:
        for lag_column in lag_columns_input:
            for i in n_days_list:
                if is_backwards == True:
                    df[f'{lag_column}(T-{i})'] = df[lag_column].shift(periods=i)
                elif is_backwards == False:
                    df[f'{lag_column}(T+{i})'] = df[lag_column].shift(periods=-i)
    return df

def _preprocess_ts(
        data: pd.DataFrame, 
        target: str, 
        n_lags_list: list, 
        n_sma_list: list, 
        fourier_order: int=1,
        dropna: bool=True,
    ) -> pd.DataFrame:
    """
    Preparing input (date, target) data for training autoregressive predictive model with lags and Fourier features.
    """
    # numerical features SMAs and daily lags
    res = _create_daily_lags(df=data, lag_columns_input=[target], n_days_list=n_lags_list, continuous=False)
    for sma_period in n_sma_list:
        sma_name = f'SMA_{sma_period}d'
        res[sma_name] = data[target].shift(1).rolling(window=sma_period, min_periods=sma_period).mean()
    # date categorical features
    res['is_weekend'] = np.select([(res.date.dt.weekday == 5) | (res.date.dt.weekday == 6)], [1], 0)
    # date Fourier numerical features
    cal_fourier_gen = CalendarFourier("y", fourier_order)
    res = res.merge(cal_fourier_gen.in_sample(res['date']), left_on='date', right_on='date')
    cal_fourier_gen = CalendarFourier("m", fourier_order)
    res = res.merge(cal_fourier_gen.in_sample(res['date']), left_on='date', right_on='date')
    cal_fourier_gen = CalendarFourier("w", fourier_order)
    res = res.merge(cal_fourier_gen.in_sample(res['date']), left_on='date', right_on='date')
    if dropna:
        res = res.dropna().reset_index(drop=True)
    return res

def _train_catboost_model(
        data: pd.DataFrame,
        target_name: str, 
        categorical_columns=[str],
        n_trials=15,
        loss='RMSE', 
        verbose=False,
        show_descent_graph=False,
    ) -> CatBoostRegressor:
    """
    Returns a trained CatBoostRegressor model.
    """

    if verbose == False:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    elif verbose == True:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        raise Exception('Verbose parameter can only be True or False')

    # Splitting data into train, val, test
    # Setting threshholds for time series split
    unique_dates = sorted(data.date.unique())
    train_threshold =  unique_dates[int(len(unique_dates) * 0.70)] #70%
    val_threshold = unique_dates[int(len(unique_dates) * 0.85)] #15%

    # Split data
    train_data = data.loc[data.date <= train_threshold]
    val_data = data.loc[(data.date > train_threshold) & (data.date <= val_threshold)]
    test_data = data.loc[data.date > val_threshold]

    # Create X, y
    X_train, y_train = train_data.drop(columns=['date', target_name]), train_data[target_name]
    X_val, y_val = val_data.drop(columns=['date', target_name]), val_data[target_name]
    X_test, y_test = test_data.drop(columns=['date', target_name]), test_data[target_name]

    # Optuna GPU hyperparameters optimizer
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1500),
            'depth': trial.suggest_int('depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'loss_function': loss,
            'task_type': 'GPU'
        }
        # Initialize and train CatBoostRegressor
        model = CatBoostRegressor(**params, cat_features=categorical_columns, verbose=0)
        model.fit(X_train, y_train, 
                    eval_set=[(X_val, y_val)], 
                    early_stopping_rounds=30,
                    verbose=False)
        # Evaluate the model on test set
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        return rmse
        
    # Initialize Optuna study and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    if show_descent_graph:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()

    # Best trial result
    if verbose:
        print(f"-- Best trial: RMSE = {study.best_value}, params = {study.best_params} --")

    best_params = study.best_params
    best_params['task_type'] = 'GPU'
    best_model = CatBoostRegressor(**best_params, 
                                   cat_features=categorical_columns,
                                   verbose=0, 
                                   random_seed=42)
    best_model.fit(X_train, y_train, 
                    eval_set=[(X_val, y_val)], 
                    early_stopping_rounds=30,
                    verbose=False)
    return best_model

def _get_slice(
        data: pd.DataFrame,
        start: float,
        end: float
    ) -> pd.DataFrame:
    unique_dates = sorted(data.date.unique())
    threshold_min =  unique_dates[int(len(unique_dates) * start)]
    threshold_max =  unique_dates[int(len(unique_dates) * end)]
    sliced_data = data.loc[(data.date >= threshold_min) & (data.date <= threshold_max)].reset_index(drop=True)
    return sliced_data

def _add_lag_features(
        input_df:pd.DataFrame,
        data:pd.DataFrame, 
        ewma_length:int, 
        n_lags_list:list, 
        prophet_model: Prophet,
    ) -> pd.DataFrame:
    """
    This function adds columns of lags. Parallel computation is not supported.
    """
    for lag_number in n_lags_list:
        lag_number = lag_number - 1
        lag_feature_name = f"cons(T-{lag_number})"
        raw_cons = data[lag_feature_name]
        lag_ewma_name = f"ewma({ewma_length})(T-{lag_number})"
        ewma_trend = data[lag_ewma_name]
        lag_date = list(data['date'] - pd.DateOffset(days=lag_number+1))
        lag_df = pd.DataFrame(data=lag_date, columns=['ds'], index=list(range(0, len(lag_date))))
        prophet_seasonality = prophet_model.predict(lag_df)['additive_terms']
        residual = raw_cons - ewma_trend - prophet_seasonality
        input_df[f'residual(T-{lag_number+1})'] = residual.values
    return input_df

def _add_sma_features(
        input_df:pd.DataFrame,
        data:pd.DataFrame, 
        feature_name:str, 
        n_sma_list:list
    ) -> pd.DataFrame:
    """
    This function adds columns of SMA of subsamples of lags.\n
    Returns original dataset with added sma_n columns.
    """
    for sma_length in n_sma_list:
        temp_df = pd.DataFrame()
        for k in range(0, sma_length):
            feature_name_index = f"{feature_name}(T-{k+1})" #format: residual(T-1)
            temp_df[feature_name_index] = data[feature_name_index]
        sma_value = temp_df.mean(axis=1)
        input_df[f"SMA_{sma_length}d"] = sma_value
    return input_df

def _add_date_features(
        input_df: pd.DataFrame,
        data: pd.DataFrame,
        fourier_order: int,
        timedelta_days: int=1,
    ) -> pd.DataFrame:
    """
    This function adds date feature columns to the left of existing single row in input dataframe.\n
    Returns pandas dataframe with added date features.
    """

    df_to_add = data[['date']]
    df_to_add['date'] = df_to_add['date'] + pd.Timedelta(days=timedelta_days)

    # Weekend - Sat, Sun
    df_to_add['is_weekend'] = np.select([(df_to_add.date.dt.weekday == 5) | (df_to_add.date.dt.weekday == 6)], [1], 0)

    # Date Fourier numerical features
    cal_fourier_gen = CalendarFourier("y", fourier_order)
    df_to_add = df_to_add.merge(cal_fourier_gen.in_sample(df_to_add['date']), left_on='date', right_on='date')
    cal_fourier_gen = CalendarFourier("m", fourier_order)
    df_to_add = df_to_add.merge(cal_fourier_gen.in_sample(df_to_add['date']), left_on='date', right_on='date')
    cal_fourier_gen = CalendarFourier("w", fourier_order)
    df_to_add = df_to_add.merge(cal_fourier_gen.in_sample(df_to_add['date']), left_on='date', right_on='date')
    
    # adding day o input dataframe
    df_to_add = df_to_add.drop(columns=['date'])
    input_df = pd.concat([input_df, df_to_add], axis=1)
    return input_df


# -----------------------------------------------------------------------
# Model class


class Model:
    """
    Key class in TeremokTSLib. Model is a packaged ensemble of Prophet, CatBoost and other methods for training on daily consumption data.
    """

    def __init__(
            self, 
    )-> None:
        self.prophet_model = Prophet(),
        self.catboost_model = CatBoostRegressor(),
        self.beta_coefficient = float(),
        self.inference_cols = list(),
        self.is_trained = False,
        self.train_date = str(),
        self.n_lags_list = list(),
        self.n_sma_list = list(),
        self.ewma_length = int,
        self.fourier_order = int,
        self.safe_stock_teta = float,
        self.regularization_gamma = float,
        self.mean_cons = float

    def __str__(self) -> str:
        return '<TeremokTSLib Model>'

    def __repr__(self) -> str:
        return '<TeremokTSLib Model>'
    
    def train(
            self,
            data: pd.DataFrame,
            k_coefficient: float=1,
            days_till_perish: int=2,
            goods_cost: float=1.0,
            alpha_coefficient: float=4.0,
            catboost_params: dict={},
            prophet_params: dict={},
            ewma_length: int=10,
            interpolate_spaces: bool=True,
            holidays_df: pd.DataFrame=make_holidays_df(year_list=[1990 + i for i in range(40)], country='RU'),
            n_lags_list: list=[1,2,3],
            n_sma_list: list=[2,3],
            fourier_order: int=3,
            optuna_trials: int=15,
            beta: list=[1.0, 2.0],
            teta: list=[0.4, 0.8],
            gamma: list=[0.05, 0.20],
            fb_njobs: int=1,
            fb_tuning: bool=False,
            show_residuals_pacf: int=0,
            show_descent_graph: bool=False,
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
        
        if not isinstance(catboost_params, dict):
            raise ValueError(f"catboost_params must be a list. Provided type: {type(catboost_params)}")
        
        if not isinstance(prophet_params, dict):
            raise ValueError(f"prophet_params must be a list. Provided type: {type(prophet_params)}")
        
        if not isinstance(ewma_length, int):
            raise ValueError(f"ewma_length must be an integer. Provided type: {type(ewma_length)}")
        
        if not isinstance(interpolate_spaces, bool):
            raise ValueError(f"interpolate_spaces must be a boolean. Provided type: {type(interpolate_spaces)}")
        
        if not isinstance(show_residuals_pacf, int):
            raise ValueError(f"show_residuals_pacf must be a integer. Provided type: {type(show_residuals_pacf)}")
        
        if not isinstance(fb_njobs, int):
            raise ValueError(f"fb_njobs must be a integer. Provided type: {type(fb_njobs)}")
        
        if not isinstance(show_descent_graph, bool):
            raise ValueError(f"show_descent_graph must be a boolean. Provided type: {type(show_descent_graph)}")
        
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
        self.n_lags_list = n_lags_list
        self.n_sma_list = n_sma_list
        self.ewma_length = ewma_length
        self.fourier_order = fourier_order


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
            print('Interpolation finished successfully!')
        elif interpolate_spaces == False and is_continuous == False:
            raise Warning("Interpolation was switched off! Data should be continuous for correct training process.")


        # calculating ewma (base layer) as Trend layer
        data['trend'] = _calc_rolling_ewma(data['consumption'], window_size=ewma_length).shift(-1)
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)


        # getting first residual and training Prophet model (first layer) as Seasonality layer
        print('Started training Prophet model...')
        fb_model = _train_prophet_model(data=data,
                                        date_col='date',
                                        target_col='consumption',
                                        trend_col='trend',
                                        holidays=holidays_df,
                                        is_tuning=fb_tuning,
                                        fb_njobs=fb_njobs)
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
        ts_for_training_cb = _preprocess_ts(data=ts_for_training, 
                                         target='residual',
                                         n_lags_list=n_lags_list,
                                         n_sma_list=n_sma_list,
                                         fourier_order=fourier_order)


        # training CatBoost model (second layer) as Residual layer
        print('Started training CatBoost model...')
        cb_model = _train_catboost_model(data=ts_for_training_cb, 
                                         target_name='residual', 
                                         categorical_columns=['is_weekend'],
                                         n_trials=optuna_trials,
                                         loss='RMSE',
                                         show_descent_graph=show_descent_graph)
        self.catboost_model = cb_model
        selected_inf_cols = []
        selected_inf_cols.extend(['date'])
        selected_inf_cols.append(f'ewma({self.ewma_length})(T+1)')
        for i in self.n_lags_list:
            selected_inf_cols.append(f'ewma({self.ewma_length})(T-{i-1})')
            selected_inf_cols.append(f'cons(T-{i-1})')
        self.inference_cols = selected_inf_cols
        self.is_trained = True
        self.train_date = datetime.now()


        # beta coefficient optimization
        validation_dataset = _get_slice(data, start=0.70, end=0.95)
        self.mean_cons = np.mean(validation_dataset['consumption'])
        validation_dataset.rename(columns={'consumption': 'cons'}, inplace=True)
        validation_cb_dataset = _preprocess_ts(data=ts_for_training, 
                                                target='residual',
                                                n_lags_list=n_lags_list,
                                                n_sma_list=n_sma_list,
                                                fourier_order=fourier_order,
                                                dropna=False)
        validation_cb_dataset = _get_slice(validation_cb_dataset, start=0.70, end=0.95)
        validation_cb_dataset.drop(columns=['date', 'residual'], inplace=True)
        validation_dataset['cons_pred'] = validation_dataset['trend'] + validation_dataset['seasonality'] + cb_model.predict(validation_cb_dataset)
        validation_dataset.dropna(inplace=True)
        initial_stock_assumption = np.percentile(validation_dataset['cons'], 85) * 1.5 #TODO: assert this later
        best_coefs = optimization._validate(data=validation_dataset,
                                           initial_stock=initial_stock_assumption,
                                           ewma_length=ewma_length,
                                           k=k_coefficient,
                                           valid_mean_cons=self.mean_cons,
                                           lifetime_d=days_till_perish,
                                           cost=goods_cost,
                                           alpha=alpha_coefficient,
                                           max_beta=max(beta),
                                           min_beta=min(beta),
                                           max_teta=max(teta),
                                           min_teta=min(teta),
                                           max_gamma=max(gamma),
                                           min_gamma=min(gamma))
        self.beta_coefficient = best_coefs["best_beta"]
        self.safe_stock_teta = best_coefs["best_teta"]
        self.regularization_gamma = best_coefs["best_gamma"]
        print('Training finished successfully!')


    def predict_consumption(
            self,
            data: pd.DataFrame,
    ) -> list:
        """
        Returns predicted consumption for each row for day T+1 (tomorrow).
        """

        # checking if all input columns are present
        for needed_col in self.inference_cols:
            if needed_col not in data.columns:
                usercase = ', '.join(self.inference_cols)
                raise ValueError(f"Input data must contain all necessary columns for inference. In your case: {usercase}")
        
        output = []
        # trend layer
        trend = data[f'ewma({self.ewma_length})(T+1)'] #TODO: - remove ewma trend (or substitute)
        # seasonality layer
        data['date'] = pd.to_datetime(data['date']) #TODO: rework this later
        tmr_dates = list(data['date'] + pd.DateOffset(days=1))
        fb_inference_data = pd.DataFrame(data=tmr_dates, columns=['ds'], index=list(range(0, data.shape[0])))
        seasonality = self.prophet_model.predict(fb_inference_data)
        seasonality = seasonality['additive_terms']
        # residual layer
        cb_inference_df = pd.DataFrame()
        cb_inference_df = _add_lag_features(input_df=cb_inference_df,
                                            data=data,
                                            ewma_length=self.ewma_length,
                                            n_lags_list=self.n_lags_list,
                                            prophet_model=self.prophet_model)
        cb_inference_df = _add_sma_features(input_df=cb_inference_df,
                                            data=cb_inference_df,
                                            feature_name='residual',
                                            n_sma_list=self.n_sma_list)
        cb_inference_df = _add_date_features(input_df=cb_inference_df,
                                             data=data,
                                             fourier_order=self.fourier_order,
                                             timedelta_days=1)
        residual = self.catboost_model.predict(cb_inference_df)
        # final output calculation
        output = trend + seasonality + residual
        return np.round(np.array(output), 2)


    def predict_order(
            self,
            data: pd.DataFrame,
    ) -> dict:
        """
        Returns recommended orders and predicted consumption as dict with keys 'order' and 'cons_pred' on day T+1 (prediction day) 
        based on forcasted consumption on day T+1.
        Note: input data should contain all necessary columns and stock_left column. 
        To view them, use view_inference_cols() method.

        Examples
        --------
        >>> model = tts.Model()
        >>> model.train(data=train_data)
        >>> order = model.predict_order(data=inference_data)
        """

        # checking if all input columns are present
        must_have_cols = ['stock_left', 'k_coef']
        must_have_cols.extend(self.inference_cols)
        for needed_col in must_have_cols:
            if needed_col not in data.columns:
                usercase = ', '.join(must_have_cols)
                raise ValueError(f"Input data must contain all necessary columns for inference. In your case: {usercase}")
            
        # getting consumption prediction 
        cons_pred = np.array(self.predict_consumption(data=data))
        cons_pred[cons_pred < 0] = 0 # preventer of boosing negative output
        # converting predicted consumption to order in boxes with k and beta coefficient
        scaling_factor = np.minimum(1, 1 - self.regularization_gamma * (cons_pred / self.mean_cons - 1))
        ewma_cons = np.array(data[f"ewma({self.ewma_length})(T+1)"])
        y_adj = np.maximum((cons_pred + self.safe_stock_teta * ewma_cons) * scaling_factor, (cons_pred * self.beta_coefficient * scaling_factor))
        y_adj_to_order = np.round(y_adj - np.array(data["stock_left"])) 
        y_adj_to_order[y_adj_to_order < 0] = 0 # preventer of negative order if some server data mistake occurs
        y_box_adj_to_order = np.round(y_adj_to_order / np.array(data["k_coef"]))
        return {"order": y_box_adj_to_order, "cons_pred": cons_pred}


    def view_cb_inference_cols(
            self,
    ) -> list:
        """
        Returns a list of necessary columns for inference of only CatBoost model. 
        Note: for Prophet only date and consumption columns are needed.
        """
        return self.inference_cols
    

    def _stresstest(
            self, 
            init_stock_left: float,
            data: pd.DataFrame,
            plot: bool=True,
    ) -> pd.DataFrame:
        """
        Simulates the workflow of automatic order on given data. Iteratively calculates orders and stocks on the spot. 
        Returns dataframe with results and (optionally) a plot.
        """

        # calculate cons and ewma for each day from data (date, consumption)
        prepd_data = _calculate_lagged_consumption_and_ewma(df=data, n_lags_list=self.n_lags_list, ewma_length=self.ewma_length)
        real_consumption = prepd_data['consumption']
        prepd_data.drop(columns=['consumption'], inplace=True)
        prepd_data.dropna(inplace=True)

        # start from init_stock_left and create orders
        return "Work in progress. Method is not realised yet."


    def itertest(
            self,
            test_data: pd.DataFrame,
            initial_stock_left: float,
            k_coefficient: int=1,
            lifetime_d: int=2,
            cost: float=1.0,
            alpha: float=4.0,
            save_name: str='new_model',
            save_results: bool=False, 
            plot: bool=True
    ) -> list:
        """
        Simulates the workflow of automatic order on given data. Iteratively calculates orders and stocks on the spot. 
        Returns dataframe with results and (optionally) a plot.        
        """

        # calculate cons and ewma for each day from data (date, consumption)
        prepd_data = _calculate_lagged_consumption_and_ewma(df=test_data, n_lags_list=self.n_lags_list, ewma_length=self.ewma_length)
        prepd_data.dropna(inplace=True)
        prepd_data.reset_index(drop=True, inplace=True)
        real_consumption = list(prepd_data['consumption'])
        prepd_data.drop(columns=['consumption'], inplace=True)

        # create predictions of consumption for each day
        if self.is_trained:
            cons_pred = self.predict_consumption(data=prepd_data)
        else:
            raise Exception("Your model is not trained. Train it first, then use itertest method.")

        # create opt_test_data = (date, cons, cons_pred)
        opt_test_data = pd.DataFrame()
        opt_test_data['date'] = prepd_data['date']
        opt_test_data['cons'] = real_consumption
        opt_test_data['cons_pred'] = cons_pred

        # proceeding to itertest
        model = optimization.EnsembleModel(k=k_coefficient, 
                                            lifetime_d=lifetime_d, 
                                            beta=self.beta_coefficient,
                                            teta=self.safe_stock_teta,
                                            gamma=self.regularization_gamma,
                                            valid_mean_cons=self.mean_cons,
                                            alpha=alpha,
                                            cost=cost,
                                            name=save_name)
        write_off, stop_sale, loss, model_order_q, sum_loss = optimization._simulate(data=opt_test_data, 
                                                                                        model=model, 
                                                                                        initial_stock=initial_stock_left, 
                                                                                        ewma_length=self.ewma_length, 
                                                                                        save_results=save_results, 
                                                                                        plot=plot)
        return [write_off, stop_sale, loss, model_order_q, sum_loss]


    def save_model(
            self, 
            model_name: str='new_model',
            foulder_name: str='saved_models',
    ) -> None:
        """
        Saves selected ensemble model in pickle format.
        """
        if not os.path.exists(foulder_name):
            os.mkdir(foulder_name)
            with open(f'{foulder_name}/{model_name}.pkl', 'wb') as file:
                pickle.dump(self, file)
        else:
            with open(f'{foulder_name}/{model_name}.pkl', 'wb') as file:
                pickle.dump(self, file)       