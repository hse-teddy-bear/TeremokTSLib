"""
Optuna optimization
---------
Framework for finding optimal beta coefficients (for convertion from consumption prediction to order).
Using Optuna we adjust our function so that the loss is minimal on validation set.
"""


# -----------------------------------------------------------------------
# dependencies


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import optuna


# -----------------------------------------------------------------------
# Simulation Order classes


class OrderModel:

    def __init__(self, 
                 k: int, 
                 lifetime_d: int, 
                 cost: float, 
                 alpha: float,
        ) -> None:
        self.k = k
        self.lifetime_d = lifetime_d
        self.cost = cost
        self.alpha = alpha

    def order(self, 
              data: pd.DataFrame, 
              day, 
              mean_cons
        ) -> int:
        model_stock_end = data.loc[day, 'model_stock_end']
        next_stock = self.next_stock(data, day, mean_cons)
        return int(round((next_stock - model_stock_end) / self.k)) * self.k
    
    #TODO: edit this later
    def next_stock(self, data, day):
        raise NotImplementedError()
    

class EnsembleModel(OrderModel):

    def __init__(self, 
                 k: int, 
                 lifetime_d: int,
                 beta: float,
                 alpha: float=1.0,
                 cost: float=1.0,
                 name: str='new_model',
        ) -> None:
        super().__init__(k, lifetime_d, cost, alpha)
        self.beta = beta
        self.name = name

    def next_stock(self, data, day, mean_cons):
        cons_pred = max(data.loc[day + 1, 'cons_pred'], 0)
        return max((cons_pred + 0.85 * mean_cons), (cons_pred * self.beta * (1 - ((cons_pred - mean_cons) / mean_cons) ** 2)))


# -----------------------------------------------------------------------
# main methods


def _prep_data(
        data: pd.DataFrame
    ) -> pd.DataFrame:
    data.reset_index(drop=True, inplace=True)
    data[['model_stock_beg', 'model_stock_end', 'order', 'write_off', 'stop_sale']] = 0
    return data

def _plot_results(
        data, 
        model, 
        write_off, 
        stop_sale, 
        save_results,
    ) -> None:
    plt.figure(figsize=(20,10))
    plt.plot(data['date'], data['cons'], label='Real consumption', color='blue')
    plt.plot(data['date'], data['cons_pred'], label='Predicted consumption', linestyle='dashed', color='red')
    plt.plot(data['date'], data['model_stock_beg'], label='Model stock', color='green')
    plt.bar(data['date'], data['write_off'], label='Write offs', color='orange')
    plt.bar(data['date'], data['stop_sale'], label='Sale stops', color='red')
    plt.hlines(y=0, xmin=data.date.min(), xmax=data.date.max(), color='grey', linewidth=2, linestyle='dashed')
    plt.legend()
    plt.suptitle(f'Itertest results on given data, {model.name}', fontsize=18)
    plt.title(f'Stop sales: {stop_sale}; Write offs: {write_off}\n' + f'beta: {model.beta}', fontsize=18, loc='right')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('stock(beggining of the day) - cons(day)', fontsize=14)
    if save_results:
        if not os.path.exists(f"results"):
            os.mkdir(f"results")
        plt.savefig(f"results/{model.name}.png")
    plt.show()

def _simulate(
        data: pd.DataFrame,
        model: EnsembleModel, 
        initial_stock: float,
        ewma_length: int,
        save_results: bool=False, 
        plot: bool=True,
    ) -> tuple:

    # prepare dataframe
    data = _prep_data(data=data)

    # set first day model stock as it was in reality
    data.loc[0, 'model_stock_beg'] = initial_stock

    test_period = list(range(0, data.shape[0]-1))
    for day in test_period:
        today_data = data.loc[day]
        stock_left = today_data['model_stock_beg'] - today_data['cons']

        stop_sale, write_off = 0, 0

        if stock_left < 0:
            stop_sale = -stock_left
            stock_left = 0
        
        if day-model.lifetime_d >= 0 and data.loc[day-model.lifetime_d, 'model_stock_beg'] - data.loc[day-model.lifetime_d:day, 'cons'].sum() - data.loc[day-model.lifetime_d:day, 'write_off'].sum() > 0:
            write_off = data.loc[day-model.lifetime_d, 'model_stock_beg'] - sum(data.loc[day-model.lifetime_d:day, 'cons']) - sum(data.loc[day-model.lifetime_d:day, 'write_off'])
            stock_left -= write_off
        
        data.loc[day, 'model_stock_end'] = stock_left
        mean_cons = np.mean(data.loc[day-ewma_length:day, 'cons'])
        order = model.order(data, day, mean_cons)
        data.loc[day+1, 'model_stock_beg'] = stock_left + order
        data.loc[day, 'order'] = order
        data.loc[day, 'write_off'] = write_off
        data.loc[day, 'stop_sale'] = stop_sale

    write_off, stop_sale = round(data['write_off'].sum()), round(data['stop_sale'].sum())
    model_order_q = round(len(test_period) / data[data['order'] > 0].shape[0], 2)
    loss = model.cost * (model.alpha * stop_sale + write_off)
    sum_cons = data['cons'].sum()
    if plot:
        _plot_results(data, model, write_off, stop_sale, save_results)
    return write_off, stop_sale, loss, model_order_q, sum_cons

def _validate(data: pd.DataFrame,
              initial_stock: float,
              ewma_length: int,
              k: int,
              lifetime_d: int,
              cost: float,
              alpha: float,
              min_beta: float=1, 
              max_beta: float=2.5,
              n_trials: int=100,
    ) -> float:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        beta = trial.suggest_float('beta', min_beta, max_beta)
        test_model = EnsembleModel(k, lifetime_d, cost, alpha, beta)
        test_model.alpha = 1
        _, _, loss, _, _ = _simulate(data, test_model, initial_stock, ewma_length, save_results=False, plot=False)
        return loss
    study = optuna.create_study(direction='minimize') #, sampler=optuna.samplers.TPESampler(seed=42)
    study.optimize(objective, n_trials=n_trials)
    best_betas = sorted([best_trial.params['beta'] for best_trial in study.best_trials])
    best_beta = np.round(np.mean(best_betas), 3)
    return best_beta

def _itertest(data,
              k: int,
              lifetime_d: int,
              cost: float,
              alpha: float,
              initial_stock_val: float,
              initial_stock_test: float,
              ewma_length: int,  
              min_beta: float=1,
              max_beta: float=1.7,
              n_trials: int=100,       
              save_results=False, 
              plot=True
    ) -> list:
    best_beta = None
    unique_dates = sorted(data.date.unique())
    valid_threshold_min =  unique_dates[int(len(unique_dates) * 0.70)]
    valid_threshold_max =  unique_dates[int(len(unique_dates) * 0.85)]
    valid_data = data.loc[(data.date >= valid_threshold_min) & (data.date <= valid_threshold_max)].reset_index()
    test_data = data.loc[data.date > valid_threshold_max].reset_index()
    best_beta = _validate(data=valid_data,
                          initial_stock=initial_stock_val,
                          ewma_length=ewma_length,
                          k=k,
                          lifetime_d=lifetime_d,
                          cost=cost,
                          alpha=alpha,
                          min_beta=min_beta,
                          max_beta=max_beta,
                          n_trials=n_trials)
    model = EnsembleModel(k, lifetime_d, cost, alpha, best_beta)
    write_off, stop_sale, loss, model_order_q, sum_loss = _simulate(data=test_data, 
                                                                    model=model, 
                                                                    initial_stock=initial_stock_test, 
                                                                    ewma_length=ewma_length, 
                                                                    save_results=save_results, 
                                                                    plot=plot)
    return [write_off, stop_sale, loss, best_beta, model_order_q, sum_loss]
    
