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
                 teta: float,
                 gamma: float,
                 valid_mean_cons: float,
                 alpha: float=1.0,
                 cost: float=1.0,
                 name: str='new_model',
        ) -> None:
        super().__init__(k, lifetime_d, cost, alpha)
        self.beta = beta
        self.gamma = gamma
        self.teta = teta
        self.name = name
        self.valid_mean_cons = valid_mean_cons

    def next_stock(self, data, day, mean_cons):
        cons_pred = max(data.loc[day + 1, 'cons_pred'], 0)
        scaling_factor = np.minimum(1, 1 - self.gamma * (cons_pred / self.valid_mean_cons - 1))
        return max((cons_pred + self.teta * mean_cons) * scaling_factor, (cons_pred * self.beta * scaling_factor))


# -----------------------------------------------------------------------
# main methods


def _prep_data(
        data: pd.DataFrame
    ) -> pd.DataFrame:
    data.reset_index(drop=True, inplace=True)
    data[['model_stock_beg', 'model_stock_end', 'order', 'write_off', 'stop_sale']] = 0
    return data

def _plot_results(
        data: pd.DataFrame, 
        model: EnsembleModel, 
        write_off, 
        stop_sale, 
        save_results: bool=False,
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
    wape = np.round(np.sum(np.abs(data['cons'] - data['cons_pred'])) / np.sum(np.abs(data['cons'])) * 100, 1)
    plt.title(f'Stop sales: {stop_sale}; Write offs: {write_off}\n' + f'beta: {model.beta}, teta_safe: {model.teta}, gamma_reg: {model.gamma}\n' + f'wape: {wape}%', fontsize=18, loc='right')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('stock(beggining of the day) - cons(day)', fontsize=14)
    if save_results:
        if not os.path.exists(f"results"):
            os.mkdir(f"results")
        plt.savefig(f"results/{model.name}.png")
        plt.close()
    else:
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
        
        if day-model.lifetime_d+1 >= 0:
            write_off_calc = data.loc[day-model.lifetime_d+1, 'model_stock_beg'] - data.loc[day-model.lifetime_d+1:day, 'cons'].sum() - data.loc[day-model.lifetime_d+1:day, 'write_off'].sum()
            if write_off_calc > 0:
                write_off = write_off_calc
                stock_left -= write_off
        
        data.loc[day, 'model_stock_end'] = stock_left
        mean_cons = np.mean(data.loc[day-ewma_length:day, 'cons'])
        order = max(0, model.order(data, day, mean_cons))
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
              valid_mean_cons: float,
              cost: float,
              alpha: float=4,
              min_beta: float=1, 
              max_beta: float=2.5,
              min_teta: float=0.4,
              max_teta: float=0.8,
              min_gamma: float=0.05,
              max_gamma: float=0.2,
              n_trials: int=200,
    ) -> float:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    beta_values = np.arange(min_beta, max_beta + 0.05, 0.05).tolist()
    teta_values = np.arange(min_teta, max_teta + 0.05, 0.05).tolist()
    gamma_values = np.arange(min_gamma, max_gamma + 0.05, 0.05).tolist()

    def objective(trial):
        _beta = trial.suggest_float('beta', min_beta, max_beta)
        _teta = trial.suggest_float('teta', min_teta, max_teta)
        _gamma = trial.suggest_float('gamma', min_gamma, max_gamma)
        test_model = EnsembleModel(k=k, 
                                   lifetime_d=lifetime_d, 
                                   beta=_beta, 
                                   teta=_teta,
                                   gamma=_gamma,
                                   valid_mean_cons=valid_mean_cons,
                                   alpha=alpha, 
                                   cost=cost)
        _, _, loss, _, _ = _simulate(data=data, 
                                     model=test_model, 
                                     initial_stock=initial_stock, 
                                     ewma_length=ewma_length, 
                                     save_results=False, 
                                     plot=False)
        return loss
    
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.GridSampler(
                                search_space={
                                    'beta': beta_values,
                                    'teta': teta_values,
                                    'gamma': gamma_values}))
    study.optimize(objective, n_trials=n_trials)
    best_coefs_dict = {}
    for coef in ["beta", "teta", "gamma"]:
        best_coefs = sorted([best_trial.params[f'{coef}'] for best_trial in study.best_trials])
        best_coef = np.round(np.mean(best_coefs), 3)
        best_coefs_dict[f"best_{coef}"] = best_coef
    return best_coefs_dict


    
