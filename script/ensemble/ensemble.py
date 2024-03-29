import os
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from typing import Tuple, Dict, List
from sklearn.metrics import roc_auc_score, log_loss

from script.contrastive_tree.augment import fe_new_col_name, get_retrieval_dataset
from script.loss import competition_log_loss

def import_model(config: dict) -> Dict[
        Dict[Dict[int, float], List[xgb.Booster]],
        Dict[Dict[int, float], List[lgb.Booster]],
    ]:
    #XGB CONTRASTIVE
    save_contrastive_path_xgb = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        config['NAME_CONTRASTIVE_XGB']
    )
    with open(
        os.path.join(
            save_contrastive_path_xgb,
            'best_result_xgb.txt'
        ), 'r'
    ) as file:
        best_result_xgb_contrastive = json.load(file)

    with open(
        os.path.join(
            save_contrastive_path_xgb,
            'model_list_xgb.pkl'
        ), 'rb'
    ) as file:
        model_list_xgb_contrastive = pickle.load(file)

    #LGB CONTRASTIVE
    save_contrastive_path_lgb = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        config['NAME_CONTRASTIVE_LGB']
    )
    with open(
        os.path.join(
            save_contrastive_path_lgb,
            'best_result_lgb.txt'
        ), 'r'
    ) as file:
        best_result_lgb_contrastive = json.load(file)

    with open(
        os.path.join(
            save_contrastive_path_lgb,
            'model_list_lgb.pkl'
        ), 'rb'
    ) as file:
        model_list_lgb_contrastive = pickle.load(file)

    #LGB
    save_path_lgb = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        config['NAME_LGB']
    )

    with open(
        os.path.join(
            save_path_lgb,
            'best_result_lgb.txt'
        ), 'r'
    ) as file:
        best_result_lgb = json.load(file)

    with open(
        os.path.join(
            save_path_lgb,
            'model_list_lgb.pkl'
        ), 'rb'
    ) as file:
        model_list_lgb = pickle.load(file)

    #XGB
    save_path_xgb = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        config['NAME_XGB']
    )

    with open(
        os.path.join(
            save_path_xgb,
            'best_result_xgb.txt'
        ), 'r'
    ) as file:
        best_result_xgb = json.load(file)

    with open(
        os.path.join(
            save_path_xgb,
            'model_list_xgb.pkl'
        ), 'rb'
    ) as file:
        model_list_xgb = pickle.load(file)


    results = {
        'lgb_contrastive': {
            'best_result': best_result_lgb_contrastive,
            'model_list': model_list_lgb_contrastive
        },
        'xgb_contrastive': {
            'best_result': best_result_xgb_contrastive,
            'model_list': model_list_xgb_contrastive
        },
        'lgb': {
            'best_result': best_result_lgb,
            'model_list': model_list_lgb
        },
        'xgb': {
            'best_result': best_result_xgb,
            'model_list': model_list_xgb
        }
    }
    return results

def xgb_contrastive_predict(
        used_feature: list,
        retrieval_dataset_0: pd.DataFrame,
        fold_: int, model_list: List[xgb.Booster], best_result: dict
    ) -> np.array:
    retrieval_dataset_0['pred'] = model_list[fold_].predict(
        xgb.DMatrix(retrieval_dataset_0[used_feature]), 
        iteration_range = (0, best_result['best_epoch'])
    )
    pred_0 = retrieval_dataset_0.groupby('rows')['pred'].median().reset_index().sort_values('rows')['pred'].values        
    return pred_0

def lgb_contrastive_predict(
        used_feature: list,
        retrieval_dataset_0: pd.DataFrame,
        fold_: int, model_list: List[lgb.Booster], best_result: dict
    ) -> np.array:
    retrieval_dataset_0['pred'] = model_list[fold_].predict(
        retrieval_dataset_0[used_feature], 
        num_iteration = best_result['best_epoch']
    )
    pred_0 = retrieval_dataset_0.groupby('rows')['pred'].median().reset_index().sort_values('rows')['pred'].values        
    return pred_0

def lgb_predict(
        used_feature: list,
        test: pd.DataFrame,
        fold_: int, model_list: List[lgb.Booster], best_result: dict
    ) -> np.array:

    test['pred'] = model_list[fold_].predict(
        test[used_feature], 
        num_iteration = best_result['best_epoch']
    )
            
    return test['pred'].values

def xgb_predict(
        used_feature: list,
        test: pd.DataFrame,
        fold_: int, model_list: List[xgb.Booster], best_result: dict
    ) -> np.array:
    test['pred'] = model_list[fold_].predict(
        xgb.DMatrix(test[used_feature]), 
        iteration_range = (0, best_result['best_epoch'])
    )
    return test['pred'].values

def get_retrieval_score(
        config_experiment: dict,
        feature_list: list, target_col: str
    ) -> None:

    model_dict = import_model(config=config_experiment)

    data = pd.read_pickle(
        os.path.join(
            config_experiment['PATH_DATA'], 
            'processed_data.pkl'
        )
    )[feature_list + ['fold', target_col]]

    log_loss_score =  0
    comp_score = 0
    used_feature = feature_list + fe_new_col_name()

    for fold_ in range(config_experiment['N_FOLD']):            
        test = data[data['fold']==fold_].reset_index(drop=True)
        
        #use for retrieval
        target_example_0 = data.loc[
            (data['fold']!=fold_) &
            (data[target_col] == 0), feature_list
        ].values
        
        test_y = test[target_col].to_numpy('float32')

        retrieval_dataset_0 = get_retrieval_dataset(test, target_example_0, feature_list, target_col)

        pred_lgb_contrastive = lgb_contrastive_predict(
            used_feature=used_feature, retrieval_dataset_0=retrieval_dataset_0, 
            fold_=fold_, model_list=model_dict['lgb_contrastive']['model_list'],
            best_result=model_dict['lgb_contrastive']['best_result']
        )
        pred_lgb = lgb_predict(
            used_feature=feature_list, test=test, fold_=fold_,
            model_list=model_dict['lgb']['model_list'],
            best_result=model_dict['lgb']['best_result']
        )
        pred_xgb_contrastive = xgb_contrastive_predict(
            used_feature=used_feature, retrieval_dataset_0=retrieval_dataset_0, 
            fold_=fold_, model_list=model_dict['xgb_contrastive']['model_list'],
            best_result=model_dict['xgb_contrastive']['best_result']
        )
        pred_xgb = xgb_predict(
            used_feature=feature_list, test=test, fold_=fold_,
            model_list=model_dict['xgb']['model_list'],
            best_result=model_dict['xgb']['best_result']
        )

        pred_1 = (pred_lgb_contrastive + pred_lgb + pred_xgb_contrastive + pred_xgb)/4

        log_loss_score += log_loss(test_y, pred_1)/config_experiment['N_FOLD']
        comp_score += competition_log_loss(test_y, pred_1)/config_experiment['N_FOLD']

    print(f'Retrieval log_loss: {log_loss_score:.5f}; Retrieval balanced log-loss: {comp_score:.5f}')