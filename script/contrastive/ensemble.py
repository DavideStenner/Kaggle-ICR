import os
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from typing import Tuple, Dict, List
from sklearn.metrics import roc_auc_score

from script.contrastive.augment import fe_pipeline, fe_new_col_name
from script.contrastive.loss import competition_log_loss

def import_model(config: dict) -> Dict[
        Dict[Dict[int, float], List[xgb.Booster]],
        Dict[Dict[int, float], List[lgb.Booster]],
    ]:

    save_path_xgb = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        config['NAME_XGB']
    )

    save_path_lgb = os.path.join(
        config['SAVE_RESULTS_PATH'], 
        config['NAME_LGB']
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

    with open(
        os.path.join(
            save_path_xgb,
            'model_list_xgb.pkl'
        ), 'rb'
    ) as file:
        model_list_xgb = pickle.load(file)

    results = {
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

def xgb_predict(
        used_feature: list,
        retrieval_dataset_0: pd.DataFrame,
        retrieval_dataset_1: pd.DataFrame,
        fold_: int, model_list: List[xgb.Booster], best_result: dict
    ) -> np.array:
        retrieval_dataset_0['pred'] = model_list[fold_].predict(
            xgb.DMatrix(retrieval_dataset_0[used_feature]), 
            iteration_range = (0, best_result['best_epoch'])
        )
        pred_0 = retrieval_dataset_0.groupby('rows')['pred'].mean().reset_index().sort_values('rows')['pred'].values

        retrieval_dataset_1['pred'] = model_list[fold_].predict(
            xgb.DMatrix(retrieval_dataset_1[used_feature]),
            iteration_range = (0, best_result['best_epoch'])
        )
        pred_1 = retrieval_dataset_1.groupby('rows')['pred'].mean().reset_index().sort_values('rows')['pred'].values
        
        pred_1 = pred_1/(pred_0+pred_1)
        return pred_1

def lgb_predict(
        used_feature: list,
        retrieval_dataset_0: pd.DataFrame,
        retrieval_dataset_1: pd.DataFrame,
        fold_: int, model_list: List[lgb.Booster], best_result: dict
    ) -> np.array:
        retrieval_dataset_0['pred'] = model_list[fold_].predict(
            retrieval_dataset_0[used_feature], 
            num_iteration = best_result['best_epoch']
        )
        pred_0 = retrieval_dataset_0.groupby('rows')['pred'].mean().reset_index().sort_values('rows')['pred'].values

        retrieval_dataset_1['pred'] = model_list[fold_].predict(
            retrieval_dataset_1[used_feature],
            num_iteration = best_result['best_epoch']
        )
        pred_1 = retrieval_dataset_1.groupby('rows')['pred'].mean().reset_index().sort_values('rows')['pred'].values
        
        pred_1 = pred_1/(pred_0+pred_1)
        return pred_1



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

    auc_ =  0
    comp_score = 0

    for fold_ in range(config_experiment['N_FOLD']):            
        test = data[data['fold']==fold_].reset_index(drop=True)
        
        #use for retrieval
        target_example_0 = data.loc[
            (data['fold']!=fold_) &
            (data[target_col] == 0), feature_list
        ].values

        target_example_1 = data.loc[
            (data['fold']!=fold_) &
            (data[target_col] == 1), feature_list
        ].values
        
        test_y = test[target_col].to_numpy('float32')

        retrieval_dataset_0 = get_retrieval_dataset(test, target_example_0, feature_list)
        retrieval_dataset_1 = get_retrieval_dataset(test, target_example_1, feature_list)

        used_feature = feature_list + fe_new_col_name()

        pred_lgb = lgb_predict(
            used_feature=used_feature, retrieval_dataset_0=retrieval_dataset_0, 
            retrieval_dataset_1=retrieval_dataset_1,
            fold_=fold_, model_list=model_dict['lgb']['model_list'],
            best_result=model_dict['lgb']['best_result']
        )

        pred_xgb = xgb_predict(
            used_feature=used_feature, retrieval_dataset_0=retrieval_dataset_0, 
            retrieval_dataset_1=retrieval_dataset_1,
            fold_=fold_, model_list=model_dict['xgb']['model_list'],
            best_result=model_dict['xgb']['best_result']
        )
        pred_1 = (pred_lgb + pred_xgb)/2

        auc_ += roc_auc_score(test_y, pred_1)/config_experiment['N_FOLD']
        comp_score += competition_log_loss(test_y, pred_1)/config_experiment['N_FOLD']

    print(f'Retrieval auc: {auc_:.5f}; Retrieval balanced log-loss: {comp_score:.5f}')

def get_retrieval_dataset(
        test: pd.DataFrame, target_example: pd.DataFrame, 
        feature_list:list
    ) -> Tuple[pd.DataFrame, list]:

    test_shape = test.shape[0]
    target_example_shape = target_example.shape[0]

    test_x = test[feature_list].to_numpy('float32')

    target_example = pd.DataFrame(
        np.concatenate(
            [
                target_example
                for _ in range(test_shape)
            ], axis=0
        ), columns=feature_list
    )

    test_x = pd.DataFrame(
        np.repeat(test_x, target_example_shape, axis=0),
        columns=feature_list
    )

    retrieval_dataset = fe_pipeline(
        dataset_1=target_example,
        dataset_2=test_x, feature_list=feature_list,
    )

    index_test = np.repeat(test.index.values, target_example_shape, axis=0)
    retrieval_dataset['rows'] = index_test

    return retrieval_dataset
