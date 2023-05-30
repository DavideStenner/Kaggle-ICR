import os
import json
import numpy as np
import pandas as pd

from scipy.optimize import Bounds, minimize
from script.loss import competition_log_loss

def get_dataset(path: str) -> np.array:
    with open(path) as config_experiment_file:
        config_experiment = json.load(config_experiment_file)
    
    pred_oof = np.load(
        os.path.join(
            config_experiment['SAVE_RESULTS_PATH'], 
            config_experiment['NAME'],
            config_experiment['OOF_FILE_NAME']
        )
    )
    return pred_oof[:, None]

if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)
    experiment_list = [
        'experiment_config_xgb.json',
        'experiment_config_lgb.json',
        'experiment_config_contrastive_lgb.json',
        'experiment_config_contrastive_lgb.json',
    ]

    oof_prediction_list = [
        get_dataset(path) for
        path in experiment_list
    ]

    target = pd.read_pickle(
        os.path.join(
            config_project['PATH_DATA'], 
            'processed_data.pkl'
        )
    )[config_project['TARGET_COL']]

    pred_array = np.concatenate(oof_prediction_list, axis=1)

    def scoring_fun(weight: np.array, prediction_array: np.array=pred_array)->float:
        pred = (prediction_array * weight[None,:]).sum(axis=1)
        return competition_log_loss(target, pred)

    constraints   = ({'type':'eq','fun':lambda w: 1 - sum(w)})

    x0 = np.array([1] * len(experiment_list))
    x = x0/len(x0)

    b = (0.0, 1.0) 
    bounds = tuple([b for x in range(len(experiment_list))])

    res = minimize(
        scoring_fun, x0,
        options={'disp': False, 'maxiter':1000},bounds = bounds,
        constraints = constraints
    )
    print(res)
    weight = res['x']/sum(res['x'])
    print(weight)