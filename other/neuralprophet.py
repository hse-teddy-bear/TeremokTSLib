"""
NeuralModel
---------
Basically we just used NeuralProphet library for this task. And added beta optimisation & itertesting. 
So, we just adopted NeuralProphet for foodtech autoorder task.
"""


# -----------------------------------------------------------------------
# dependencies


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
import os, pickle
from neuralprophet import NeuralProphet
from TeremokTSLib.itertest import optimization


# -----------------------------------------------------------------------
# support methods


def _get_slice(
        data: pd.DataFrame,
        start: float,
        end: float,
        date_col: str,
    ) -> pd.DataFrame:
    unique_dates = sorted(data[date_col].unique())
    threshold_min =  unique_dates[int(len(unique_dates) * start)]
    threshold_max =  unique_dates[int(len(unique_dates) * end)]
    sliced_data = data.loc[(data[date_col] >= threshold_min) & (data[date_col] <= threshold_max)].reset_index(drop=True)
    return sliced_data

def _add_nan_rows(
        df: pd.DataFrame, 
        starting_nans=0, 
        ending_nans=0
    ) -> pd.DataFrame:
    nan_start_df = pd.DataFrame(np.nan, index=range(starting_nans), columns=df.columns)
    nan_end_df = pd.DataFrame(np.nan, index=range(ending_nans), columns=df.columns)
    result_df = pd.concat([nan_start_df, df, nan_end_df], ignore_index=True)
    return result_df

def _get_np_inference_df(
        input_row: pd.Series, 
        n_lags: int,
        future_dates: int=1,
        start_date: str='row_date',
    ) -> pd.DataFrame:
    """
    Creates inference data for NeuralProphet model from input row with data for specified date.
    Returns long-format dataframe with columns ds, y. Future dates have NaNs in column y.
    """
    inference_data = pd.DataFrame(columns=['ds', 'y'])
    if start_date=='today':
        starting_date = datetime.today().date()
    elif start_date=='row_date':
        starting_date = input_row['date']
    else:
        starting_date = pd.to_datetime(start_date)
    for lag_index in reversed(range(0, n_lags)):
        lag_cons = input_row[f'cons(T-{lag_index})']
        lag_date = starting_date + pd.DateOffset(days=-(lag_index))
        add_date = pd.DataFrame(columns=['ds', 'y'], data=[[lag_date, lag_cons]])
        inference_data = pd.concat([inference_data, add_date], ignore_index=True, axis=0)
    for future_index in range(1, future_dates+1):
        future_date = starting_date + pd.DateOffset(days=future_index)
        add_date = pd.DataFrame(columns=['ds', 'y'], data=[[future_date, pd.NA]])
        inference_data = pd.concat([inference_data, add_date], ignore_index=True, axis=0)
    return inference_data

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
        shifted_cons = df['consumption'].shift(lag-1)
        lag_col = f'cons(T-{lag-1})'
        df[lag_col] = shifted_cons
    return df

# -----------------------------------------------------------------------
# NeuralModel


class NeuralModel:
    """
    NeuralProphet model for training on daily consumption data.
    """

    def __init__(
            self, 
    )-> None:
        self.np_model = None,
        self.beta_coefficient = float,
        self.is_trained = False,
        self.train_date = str,
        self.ewma_length = int,
        self.safe_stock_teta = float,
        self.regularization_gamma = float,
        self.mean_cons = float
        self.disable_safe_stock = False

    def __str__(self) -> str:
        return '<NeuralProphet Model>'

    def __repr__(self) -> str:
        return '<NeuralProphet Model>'
    
    def train(
            self,
            data: pd.DataFrame,
            gpu_enabled: bool=True,
            k_coefficient: float=1,
            days_till_perish: int=2,
            goods_cost: float=1.0,
            alpha_coefficient: float=4.0,
            np_model_config: dict={},
            n_lags: int=7,
            holidays_params: dict={},
            holidays_df: pd.DataFrame=pd.DataFrame({'event':'ny', 'ds':pd.to_datetime(['2024-01-01'])}),
            ewma_length: int=10,
            beta: list=[1.0, 2.0],
            teta: list=[0.4, 0.8],
            gamma: list=[0.05, 0.20],
            disable_safe_stock: bool=False,
    ) -> None:
        """
        Method for training a NeuralProphet model on given time series data.
        Note: you should pass data in correct format (See docs).

        Examples
        --------
        >>> import TeremokTSLib as tts
        >>> model = tts.NeuralModel()
        >>> model.train(data=my_df)
        """

        logging.getLogger().setLevel(logging.ERROR)

        # checking input daily data for sortedness
        data = data.sort_values(by='date').reset_index(drop=True)
        data.columns = ['ds', 'y']
        self.ewma_length = ewma_length
        self.n_lags = n_lags
        self.disable_safe_stock = disable_safe_stock

        # training single neural model
        print('Started training NeuralProphet model...')

        if gpu_enabled:
            trainer_config = {"accelerator":"gpu"}
        else:
            trainer_config = {"accelerator":"cpu"}
        model = NeuralProphet(
            **np_model_config,
            trainer_config=trainer_config
        )
        model = model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        if len(holidays_params.keys()) != 0:
            model = model.add_events(**holidays_params)
            data = model.create_df_with_events(df=data, events_df=holidays_df)
        metrics = model.fit(df=data[:-60:], 
                            freq='D',
                            validation_df=data[-60::],
                            early_stopping=50,
                            progress=None,
                            checkpointing=False,
                            continue_training=False,
                            )
        self.np_model = model
        self.is_trained = True
        self.train_date = datetime.now()

        # coefficients optimization
        validation_dataset = _get_slice(data=data, start=0.70, end=0.99, date_col='ds')
        self.mean_cons = np.mean(validation_dataset['y'])
        np_v_df = validation_dataset.copy()
        validation_dataset['cons_pred'] = self.np_model.predict(df=np_v_df, decompose=False)['yhat1'] 
        validation_dataset.rename(columns={'ds': 'date', 'y': 'cons'}, inplace=True)
        validation_dataset = validation_dataset[['date', 'cons', 'cons_pred']]
        validation_dataset.dropna(inplace=True)
        initial_stock_assumption = validation_dataset.head(1)['cons'].values[0] * 1.2
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
                                           min_gamma=min(gamma),
                                           disable_safe_stock=self.disable_safe_stock)
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
        Note: data should contain current date, ewma(N)(T+1), cons(T-t)... for n_lags specified during training.
        """
        output = []
        for row in data.iterrows():
            inference_df = _get_np_inference_df(input_row=row[1], 
                                                n_lags=self.n_lags,
                                                future_dates=1,
                                                start_date='row_date')
            prediction = self.np_model.predict(df=inference_df, decompose=False)['yhat1'].tail(1).values[0]
            output.append(prediction)
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
        >>> model = tts.NeuralModel()
        >>> model.train(data=train_data)
        >>> order = model.predict_order(data=inference_data)
        """

        # checking if all input columns are present
        must_have_cols = ['stock_left', 'k_coef']
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
        if self.disable_safe_stock:
            y_adj = (cons_pred * self.beta_coefficient * scaling_factor)
        else:
            y_adj = np.maximum((cons_pred + self.safe_stock_teta * ewma_cons) * scaling_factor, (cons_pred * self.beta_coefficient * scaling_factor))
        y_adj_to_order = np.round(y_adj - np.array(data["stock_left"])) 
        y_adj_to_order[y_adj_to_order < 0] = 0 # preventer of negative order if some server data mistake occurs
        y_box_adj_to_order = np.round(y_adj_to_order / np.array(data["k_coef"]))
        return {"order": y_box_adj_to_order, "cons_pred": cons_pred}
    

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

        # create predictions of consumption for each day
        if self.is_trained:
            inf_data = _calculate_lagged_consumption_and_ewma(df=test_data, 
                                                              n_lags_list=list(range(1, self.n_lags+1)), 
                                                              ewma_length=self.ewma_length)
            inf_data.dropna(inplace=True)
            cons_pred = self.predict_consumption(data=inf_data)
            cons_pred = [None] * max(self.ewma_length-1, self.n_lags) + list(cons_pred) + [None] * 1
        else:
            raise Exception("Your model is not trained. Train it first, then use itertest method.")

        # create opt_test_data = (date, cons, cons_pred)
        opt_test_data = pd.DataFrame()
        opt_test_data['date'] = test_data['date']
        opt_test_data['cons'] = list(test_data['consumption'])
        opt_test_data['cons'] = opt_test_data['cons'].shift(-1) # we are doing this as we compare TMR cons with predicted TMR cons
        opt_test_data['cons_pred'] = cons_pred
        opt_test_data.dropna(inplace=True)

        # proceeding to itertest
        model = optimization.EnsembleModel(k=k_coefficient, 
                                            lifetime_d=lifetime_d, 
                                            beta=self.beta_coefficient,
                                            teta=self.safe_stock_teta,
                                            gamma=self.regularization_gamma,
                                            valid_mean_cons=self.mean_cons,
                                            disabled_safe_stock=self.disable_safe_stock,
                                            alpha=alpha,
                                            cost=cost,
                                            name=save_name)
        write_off, stop_sale, loss, model_order_q, sum_loss = optimization._simulate(data=opt_test_data, 
                                                                                        model=model, 
                                                                                        initial_stock=initial_stock_left, 
                                                                                        ewma_length=self.ewma_length, 
                                                                                        save_results=save_results, 
                                                                                        plot=plot)
        return opt_test_data #[write_off, stop_sale, loss, model_order_q, sum_loss]


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