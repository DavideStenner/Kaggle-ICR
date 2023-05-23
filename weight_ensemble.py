import os
import json
import numpy as np
import pandas as pd

from scipy.optimize import Bounds, minimize
from script.contrastive.loss import competition_log_loss

if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)

    with open('experiment_config_lgb.json') as config_experiment_file:
        config_experiment_lgb = json.load(config_experiment_file)

    with open('experiment_config_xgb.json') as config_experiment_file:
        config_experiment_xgb = json.load(config_experiment_file)

    target = pd.read_pickle(
        os.path.join(
            config_project['PATH_DATA'], 
            'processed_data.pkl'
        )
    )[config_project['TARGET_COL']]

    pred_lgb = np.load(
        os.path.join(
            config_experiment_lgb['SAVE_RESULTS_PATH'], 
            config_experiment_lgb['NAME'],
            'lgb_pred_oof.npy'
        )
    )

    pred_xgb = np.load(
        os.path.join(
            config_experiment_xgb['SAVE_RESULTS_PATH'], 
            config_experiment_xgb['NAME'],
            'xgb_pred_oof.npy'
        )
    )
    pred_array = np.concatenate((pred_lgb[:, None], pred_xgb[:, None]), axis=1)

    def scoring_fun(weight: np.array, prediction_array: np.array=pred_array)->float:
        pred = (prediction_array * weight[None,:]).sum(axis=1)
        return competition_log_loss(target, pred)

    constraints   = ({'type':'eq','fun':lambda w: 1 - sum(w)})

    x0 = np.array([1, 1])
    x = x0/len(x0)

    b = (0.0, 1.0) 
    bounds = tuple([b for x in range(2)])

    res = minimize(
        scoring_fun, x0,
        options={'disp': True,'maxiter':1000},bounds = bounds,
        constraints = constraints
    )
    print(res)
    weight = res['x']/sum(res['x'])
    print(weight)