import os
import gc
import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from typing import Tuple

from sklearn.metrics import roc_auc_score, log_loss
from script.loss import competition_log_loss, calc_log_loss_weight

def get_dataset(
        data: pd.DataFrame, fold_: int, inference: bool,
        target_col: str, feature_list: list
    ) -> xgb.DMatrix:
        
    mask_fold = data['fold']==fold_ if inference else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)
    w0, w1 = calc_log_loss_weight(data[target_col])
    
    weight_ = data[target_col].map({0: w0, 1: w1})
    train_x = data[feature_list].to_numpy('float32')
    train_y = data[target_col].to_numpy('float32')

    dataset = xgb.DMatrix(train_x, train_y, weight=weight_)

    return dataset

def run_xgb_experiment(
        config_experiment: dict, params_model: dict,
        feature_list: list, target_col: str,
    ) -> None:

    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'], 
        config_experiment['NAME']
    )
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_list = []
    progress_list = []

    train = pd.read_pickle(
        os.path.join(config_experiment['PATH_DATA'], 'processed_data.pkl')
    )[feature_list + ['fold', target_col]]

    for fold_ in range(config_experiment['N_FOLD']):
        print(f'\n\nStarting fold {fold_}\n\n\n')
        
        train_matrix = get_dataset(
            data=train, fold_=fold_, inference=False, 
            target_col=target_col, feature_list=feature_list,
        )
        test_matrix = get_dataset(
            data=train, fold_=fold_, 
            inference=True, target_col=target_col, 
            feature_list=feature_list,
        )

        progress = {}

        model = xgb.train(
            params=params_model,
            dtrain=train_matrix, 
            num_boost_round=params_model['n_round'],
            evals=[(test_matrix, 'valid')], 
            verbose_eval=config_experiment['LOG_EVALUATION'],
            evals_result=progress,
        )

        if config_experiment['SAVE_MODEL']:
            model.save_model(
                os.path.join(
                       save_path,
                    f'xgb_{fold_}.json'
                )
            )

        model_list.append(model)
        progress_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()
        if config_experiment['SAVE_MODEL']:
            save_model(
                model_list=model_list, progress_list=progress_list,
                save_path=save_path
            )

def save_model(
        model_list: list, progress_list: list, save_path: str
    )->None:
        with open(
            os.path.join(
                save_path,
                'model_list_xgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)

        with open(
            os.path.join(
                save_path,
                'progress_list_xgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)

def evaluate_xgb_score(
        config_experiment: dict, 
        params_model: dict, feature_list: list,
        target_col: dict
    ) -> None:

    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'], 
        config_experiment['NAME']
    )
    
    # Find best epoch
    with open(
        os.path.join(
            save_path,
            'progress_list_xgb.pkl'
        ), 'rb'
    ) as file:
        progress_list = pickle.load(file)

    with open(
        os.path.join(
            save_path,
            'model_list_xgb.pkl'
        ), 'rb'
    ) as file:
        model_list = pickle.load(file)

        
    progress_dict = {
        'time': range(params_model['n_round']),
    }

    progress_dict.update(
            {
                f"{params_model['eval_metric']}_fold_{i}": progress_list[i]['valid'][params_model['eval_metric']]
                for i in range(config_experiment['N_FOLD'])
            }
        )

    progress_df = pd.DataFrame(progress_dict)

    progress_df[f"average_{params_model['eval_metric']}"] = progress_df.loc[
        :, [params_model['eval_metric'] in x for x in progress_df.columns]
    ].mean(axis =1)
    
    progress_df[f"std_{params_model['eval_metric']}"] = progress_df.loc[
        :, [params_model['eval_metric'] in x for x in progress_df.columns]
    ].std(axis =1)

    best_epoch = (
        int(progress_df[f"average_{params_model['eval_metric']}"].argmax())
        if config_experiment['INCREASE'] else
        int(progress_df[f"average_{params_model['eval_metric']}"].argmin())
    )
    best_score = progress_df.loc[
        best_epoch,
        f"average_{params_model['eval_metric']}"].max()
    
    std_score = progress_df.loc[
        best_epoch, f"std_{params_model['eval_metric']}"
    ]

    print(f'Best epoch: {best_epoch}, CV-log-loss: {best_score:.5f} ± {std_score:.5f}')

    best_result = {
        'best_epoch': best_epoch+1,
        'best_score': best_score
    }

    with open(
        os.path.join(
            save_path,
            'best_result_xgb.txt'
        ), 'w'
    ) as file:
        json.dump(best_result, file)

    get_retrieval_score(
        config_experiment=config_experiment,
        best_result=best_result, model_list=model_list,
        feature_list=feature_list, target_col=target_col
    )
    
def get_retrieval_score(
        config_experiment: dict,
        best_result: dict, model_list: Tuple[xgb.Booster, ...],
        feature_list: list, target_col: str
    ) -> None:
    
    data = pd.read_pickle(
        os.path.join(
            config_experiment['PATH_DATA'], 
            'processed_data.pkl'
        )
    )[feature_list + ['fold', target_col]]

    log_loss_score =  0
    comp_score = 0
    prediction_array = np.zeros((data.shape[0]))

    used_feature = feature_list

    for fold_ in range(config_experiment['N_FOLD']):            
        test = data[data['fold']==fold_].reset_index(drop=True)
        
        test['pred'] = model_list[fold_].predict(
            xgb.DMatrix(test[feature_list]), 
            iteration_range = (0, best_result['best_epoch'])
        )
        
        y_true, y_pred = test[target_col].to_numpy('float32'), test['pred']
        
        prediction_array[data['fold']==fold_] = test['pred']
        log_loss_score += log_loss(y_true, y_pred)/config_experiment['N_FOLD']
        comp_score += competition_log_loss(y_true, y_pred)/config_experiment['N_FOLD']

    print(f'log_loss: {log_loss_score:.5f}; balanced log-loss: {comp_score:.5f}')

    np.save(
        os.path.join(
            config_experiment['SAVE_RESULTS_PATH'], 
            config_experiment['NAME'],
            'xgb_pred_oof.npy'
        ),
        prediction_array
    )