"""
MinMaxModel
---------
A framework model for optimization and autoorder of long-living goods (more than 3 days).
"""


# -----------------------------------------------------------------------
# dependencies


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import optuna, os, logging
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import itertools
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from datetime import datetime
import pickle


# -----------------------------------------------------------------------
# support


def _smooth_time_series(
        values: list, 
        left_window: int, 
        right_window: int,
    ) -> list:

    if left_window < 0 or right_window < 0:
        raise ValueError("Windows should be positive integers.")
    
    smoothed_values = []
    n = len(values)
    
    for i in range(n):
        start_index = max(0, i - left_window)
        end_index = min(n, i + right_window + 1)
        window_values = values[start_index:end_index]
        smoothed_value = np.mean(window_values)
        smoothed_values.append(smoothed_value)

    return smoothed_values

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

    logger = logging.getLogger('cmdstanpy')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

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
            m.add_seasonality(name='yearly', period=365, fourier_order=4) # <- never include this fucker!
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
    model.add_seasonality(name='yearly', period=365, fourier_order=4) #<- never include this fucker!

    # Fit the model
    model.fit(train_data)
    return model

def _plot_results(
        data: pd.DataFrame, 
        write_off: float, 
        stop_sale: float, 
        save_name: str='new_trial',
        save_results: bool=False,
    ) -> None:
    
    fig = go.Figure()

    # Линия для Model stock
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['model_stock_beg'], 
        mode='lines',
        name='Model stock', 
        line=dict(color='green')
    ))

    # Линия для Real consumption
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['cons'], 
        mode='lines',
        name='Real consumption', 
        line=dict(color='blue')
    ))

    # Линия для Stock cap
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['stock_cap'], 
        mode='lines',
        name='Stock cap', 
        line=dict(color='red', dash='dash'),
        line_width=3
    ))

    # Линия для Stock floor
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['stock_floor'], 
        mode='lines',
        name='Stock floor', 
        line=dict(color='black', dash='dash'),
        line_width=3
    ))

    # Бар для Write offs
    fig.add_trace(go.Bar(
        x=data['date'], 
        y=data['write_off'], 
        name='Write offs', 
        marker=dict(color='orange')
    ))

    # Бар для Sale stops
    fig.add_trace(go.Bar(
        x=data['date'], 
        y=data['stop_sale'], 
        name='Sale stops', 
        marker=dict(color='red')
    ))

    # Горизонтальная линия на уровне y=0
    fig.add_hline(y=0, line=dict(color='grey', dash='dash', width=2))

    # Настройки заголовков и подписей
    fig.update_layout(
        title_text=f'Itertest results on given data, {save_name}',
        title_x=0.5,
        title_font_size=18,
        xaxis_title='Date',
        yaxis_title='Stock(beggining of the day) - Cons(day)',
        annotations=[dict(
            text=f'Stop sales: {stop_sale}; Write offs: {write_off}',
            xref="paper", yref="paper",
            x=1, y=1, showarrow=False,
            font=dict(size=18)
        )],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1200,
        height=600
    )

    # Сохранение или отображение графика
    if save_results:
        if not os.path.exists(f"results"):
            os.mkdir(f"results")
        #fig.write_image(f"results/{save_name}.png")
        fig.write_html(f"results/{save_name}.html")
    else:
        fig.show()

def _prep_data(
        data: pd.DataFrame
    ) -> pd.DataFrame:
    data.reset_index(drop=True, inplace=True)
    data[['model_stock_beg', 'model_stock_end', 'order', 'write_off', 'stop_sale', 'stock_cap', 'stock_floor']] = 0
    return data

def _simulate(
    data: pd.DataFrame, 
    lifetime_d: int,
    k_coef: int,
    cap_coef: float,
    floor_coef: float,
    initial_stock: float,
    cost: float=1,
    alpha: float=4,
    save_name: str='new_trial', 
    save_results=False, 
    plot=False    
    ) -> list:

    # prepare dataframe
    data = data.copy()
    data = _prep_data(data=data)
    alltime_mean_cons = np.percentile(data['cons'], 80)

    # set first day model stock as it was in reality
    data.loc[0, 'model_stock_beg'] = initial_stock

    test_period = list(range(0, data.shape[0]))
    for day in test_period:
        today_data = data.loc[day]
        stock_left = today_data['model_stock_beg'] - today_data['cons']

        stop_sale, write_off = 0, 0

        if stock_left < 0:
            stop_sale = -stock_left
            stock_left = 0
        
        if day-lifetime_d+1 >= 0:
            write_off_calc = data.loc[day-lifetime_d+1, 'model_stock_beg'] - data.loc[day-lifetime_d+1:day, 'cons'].sum() - data.loc[day-lifetime_d+1:day, 'write_off'].sum()
            if write_off_calc > 0:
                write_off = write_off_calc
                stock_left -= write_off

        # making order
        scaling_factor = np.maximum(0.8, 1 - 0.1 * (today_data['cons'] / alltime_mean_cons - 1))
        cons_pred = data.loc[day, 'cons_pred'] #data.loc[day, f'ewma({ewma_length})(T+1)'] <- we use either ewma or cons_pred as base
        stock_cap = max((floor_coef * cons_pred + cap_coef * cons_pred) * scaling_factor, k_coef)
        stock_floor = max(floor_coef * cons_pred * scaling_factor, alltime_mean_cons)
        data.loc[day, 'scaling_factor'] = scaling_factor
        if stock_left <= stock_floor:
            amt_to_replenish = stock_cap - stock_left
            boxes_to_order = max(round(amt_to_replenish / k_coef), 1)
            order = boxes_to_order * k_coef
        else:
            order = 0

        data.loc[day, 'model_stock_end'] = stock_left
        data.loc[day+1, 'model_stock_beg'] = stock_left + order
        data.loc[day+1, 'stock_cap'] = stock_cap
        data.loc[day+1, 'stock_floor'] = stock_floor
        data.loc[day, 'order'] = order
        data.loc[day, 'write_off'] = write_off
        data.loc[day, 'stop_sale'] = stop_sale

    data = data.dropna()
    write_off, stop_sale = round(data['write_off'].sum()), round(data['stop_sale'].sum())
    model_order_q = round(len(test_period) / data[data['order'] > 0].shape[0], 2)
    loss = cost * (alpha * stop_sale + write_off)
    sum_cons = data['cons'].sum()
    if plot:
        _plot_results(data=data, write_off=write_off, stop_sale=stop_sale, save_results=save_results, save_name=save_name)
    return write_off, stop_sale, loss, model_order_q, sum_cons

def _validate(
        data: pd.DataFrame,
        initial_stock: float,
        k_coef: int,
        lifetime_d: int,
        cost: float,
        alpha: float=4,
        low_cap: float=1,
        max_cap: float=4,
        low_floor: float=2,
        max_floor: float=10,
        step: float=0.5,
        n_trials: int=5,
    ) -> float:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cap_values = np.arange(low_cap, max_cap + step, step).tolist()
    floor_values = np.arange(low_floor, max_floor + step, step).tolist()
    sampler = optuna.samplers.GridSampler(search_space={'cap': cap_values,
                                                        'floor': floor_values,})

    def objective(trial):
        _cap = trial.suggest_float('cap', low=low_cap, high=max_cap)
        _floor = trial.suggest_float('floor', low=low_floor, high=max_floor)
        _, _, loss, _, _ = _simulate(data=data, 
                                     lifetime_d=lifetime_d,
                                     k_coef=k_coef,
                                     cap_coef=_cap,
                                     floor_coef=_floor,
                                     cost=cost,
                                     alpha=alpha,
                                     initial_stock=initial_stock,
                                     save_results=False, 
                                     plot=False)
        return loss
    
    study = optuna.create_study(direction='minimize',
                                sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    best_coefs_dict = {}
    best_trials = [t for t in study.trials if t.value == study.best_value]
    best_trial = min(best_trials, key=lambda t: (t.params['cap'], t.params['floor']))    
    best_coefs_dict[f"best_cap"] = best_trial.params['cap']
    best_coefs_dict[f"best_floor"] = best_trial.params['floor']  
    return best_coefs_dict

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


# -----------------------------------------------------------------------
# MinMaxModel


class MinMaxModel():
    """
    Key class in TeremokTSLib for items which take a long time to spoil.
    """

    def __init__(
            self,
    )-> None:
        self.prophet_model = Prophet(),
        self.cap_coef = float(),
        self.floor_coef = float(),
        self.inference_cols = list(),
        self.is_trained = False,
        self.train_date = str(),

    def __str__(self) -> str:
        return '<TeremokTSLib MinMaxModel>'

    def __repr__(self) -> str:
        return '<TeremokTSLib MinMaxModel>'
    
    def train(
        self,
        data: pd.DataFrame,
        k_coef: float=1,
        days_till_perish: int=2,
        goods_cost: float=1.0,
        alpha_coef: float=4.0,
        interpolate_spaces: bool=True,
        holidays_df: pd.DataFrame=make_holidays_df(year_list=[1990 + i for i in range(40)], country='RU'),
        fb_njobs: int=1,
        fb_tuning: bool=False,
        low_cap: float=1.0,
        max_cap: float=6.0,
        low_floor: float=2.0,
        max_floor: float=6.0,
        optuna_step: float=0.5,
    ) -> None:
        
        # checking input daily data for sortedness
        data = data.sort_values(by='date').reset_index(drop=True)

        # applying params
        self.k_coef = k_coef
        self.alpha_coef = alpha_coef
        self.goods_cost = goods_cost
        self.alltime_mean_cons = np.percentile(data['consumption'], 80)

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
        

        # preps for training fb prophet
        data.rename(columns={'consumption':'cons'}, inplace=True)
        data['trend'] = 0 # no trend needed
        data['smoothed_cons'] = _smooth_time_series(values = list(data['cons']), left_window=5, right_window=5) 

        # training fb prophet model
        print('Started training Prophet model...')
        fb_model = _train_prophet_model(data=data,
                                        date_col='date',
                                        target_col='smoothed_cons',
                                        trend_col='trend',
                                        holidays=holidays_df,
                                        is_tuning=fb_tuning,
                                        fb_njobs=fb_njobs)
        self.prophet_model = fb_model   
        data.drop(columns=['smoothed_cons'], inplace=True)

        # getting prophet predictions for all dataframe
        ts_data_prophet = data[['date']].rename(columns={'date':'ds'})
        prophet_predictions = self.prophet_model.predict(ts_data_prophet)

        # optimizing cap and floor coefs
        validation_dataset = data[['date', 'cons']]
        validation_dataset['cons_pred'] = prophet_predictions['yhat']
        print('Started cap/floor optimization...')
        params = _validate(data=validation_dataset,
                            lifetime_d=days_till_perish,
                            k_coef=k_coef,
                            cost=goods_cost,
                            alpha=alpha_coef,
                            initial_stock=validation_dataset.head(1)['cons'].values[0] * 2,
                            max_cap=max_cap,
                            low_cap=low_cap,
                            low_floor=low_floor,
                            max_floor=max_floor,
                            step=optuna_step,
                            n_trials=150)
        self.cap_coef = params['best_cap']
        self.floor_coef = params['best_floor']
        self.is_trained = True
        self.train_date = datetime.now()
        print('Finished successfully!')

    def predict_consumption(            
            self,
            data: pd.DataFrame,
        ) -> np.array:
        """
        Note: date column should contain TMR dates (predict day dates).
        """
        data['date'] = pd.to_datetime(data['date']) 
        dates = list(data['date'])
        fb_inference_data = pd.DataFrame(data=dates, columns=['ds'], index=list(range(0, data.shape[0])))
        cons_pred = self.prophet_model.predict(fb_inference_data)['yhat']
        cons_pred = np.round(np.array(cons_pred), 2)
        cons_pred = np.maximum(cons_pred, 0)
        return np.round(np.array(cons_pred), 2) 

    def predict_order(
            self,
            data: pd.DataFrame,
    ) -> np.array:
        """
        Note: date column should contain dates when you predict on TMR date (so the prev date to predict-date).
        """
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'] + pd.DateOffset(days=1)) 
        cons_pred = np.array(self.predict_consumption(data=data))
        cons_pred[cons_pred < 0] = 0
        data['cons_pred'] = cons_pred 

        boxes_to_order = []
        for index, row in data.iterrows():
            scaling_factor = np.maximum(0.8, 1 - 0.1 * (row['cons_pred'] / self.alltime_mean_cons - 1))
            stock_cap = max((self.floor_coef * row['cons_pred'] + self.cap_coef * row['cons_pred']) * scaling_factor, self.k_coef)
            stock_floor = max(self.floor_coef * row['cons_pred'] * scaling_factor, self.alltime_mean_cons)
            if row['stock_left'] <= stock_floor:
                amt_to_replenish = stock_cap - row['stock_left']
                order = np.maximum(np.round(amt_to_replenish / self.k_coef), 1)
            else:
                order = 0
            boxes_to_order.append(order)
        return {"order": np.array(boxes_to_order), "cons_pred": cons_pred}
        
    def itertest(
        self,
        data: pd.DataFrame,
        days_till_perish: int,
        plot: bool=True,
        save_results: bool=False,
        save_name: str='new_trial'
    ) -> list:
        
        iter_data = data.copy()
        iter_data.rename(columns={'consumption':'cons'}, inplace=True)
        initial_stock = iter_data.head(1)['cons'].values[0] * 2
        iter_data['cons_pred'] = self.predict_consumption(data=iter_data)
        iter_data['cons_pred'] = iter_data['cons_pred'].shift(-1) # we are looking at TMR cons_pred and calc cap and floor from it. And compare with todays stock_left.
        iter_data.dropna(inplace=True)

        results = _simulate(data=iter_data,
                            lifetime_d=days_till_perish,
                            k_coef=self.k_coef,
                            cap_coef=self.cap_coef,
                            floor_coef=self.floor_coef,
                            initial_stock=initial_stock,
                            cost=self.goods_cost,
                            alpha=self.alpha_coef,
                            save_results=save_results,
                            save_name=save_name,
                            plot=plot)
        return results
    
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
        