import os
import json
import pickle

import pandas as pd
import lightgbm as lgb

from typing import Tuple

from script.contrastive.augment import fe_new_col_name
from script.contrastive.lgb_model import get_retrieval_dataset

def predict_test_set(
        config_experiment: dict,
        best_result_lgb: dict, model_list_lgb: Tuple[lgb.Booster, ...],
        feature_list: list, target_col: str
    ) -> None:

    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'], 
        config_experiment['NAME']
    )
    
    with open(
        os.path.join(
            save_path,
            'model_list_lgb.pkl'
        ), 'rb'
    ) as file:
        model_list_lgb = pickle.load(file)

    with open(
        os.path.join(
            save_path,
            'best_result_lgb.txt'
        ), 'r'
    ) as file:
        best_result_lgb = json.load(file)
        
    feature_list = config_experiment['ORIGINAL_FEATURE']

    data = pd.read_pickle(
        os.path.join(
            config_experiment['PATH_DATA'], 
            'processed_data.pkl'
        )
    )[feature_list + ['fold', target_col]]

    test = pd.read_csv(
        os.path.join(
            config_experiment['PATH_DATA_ORIGINAL'],
            'test.csv'
        )
    )[feature_list]
    test['EJ'] = test['EJ'].map(
        {
            'A': 0,
            'B': 1
        }
    ).astype('uint8')

    submission = pd.read_csv(
        os.path.join(
            config_experiment['PATH_DATA_ORIGINAL'],
            'sample_submission.csv'
        )
    )
    submission['class_1'] = 0

    for fold_ in range(config_experiment['N_FOLD']):            

        #use for retrieval
        target_example_0 = data.loc[
            (data['fold']!=fold_) &
            (data[target_col] == 0), feature_list
        ].values

        target_example_1 = data.loc[
            (data['fold']!=fold_) &
            (data[target_col] == 1), feature_list
        ].values
        
        retrieval_dataset_0 = get_retrieval_dataset(test, target_example_0, feature_list)
        retrieval_dataset_1 = get_retrieval_dataset(test, target_example_1, feature_list)

        used_feature = feature_list + fe_new_col_name()

        retrieval_dataset_0['pred'] = model_list_lgb[fold_].predict(
            retrieval_dataset_0[used_feature], 
            num_iteration = best_result_lgb['best_epoch']
        )
        pred_0 = retrieval_dataset_0.groupby('rows')['pred'].mean().reset_index().sort_values('rows')['pred'].values

        retrieval_dataset_1['pred'] = model_list_lgb[fold_].predict(
            retrieval_dataset_1[used_feature],
            num_iteration = best_result_lgb['best_epoch']
        )
        pred_1 = retrieval_dataset_1.groupby('rows')['pred'].mean().reset_index().sort_values('rows')['pred'].values
        
        pred_1 = pred_1/(pred_0+pred_1)
        
        submission['class_1'] += pred_1/config_experiment['N_FOLD']

    submission['class_0'] = 1- submission['class_1']
    submission.to_csv('submission.csv', index=False)



if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)

    feature_list = config_project['ORIGINAL_FEATURE'] + fe_new_col_name()
    predict_test_set(
        config_experiment=config_project, 
        best_result_lgb=best_result_lgb, model_list_lgb=model_list_lgb, 
        feature_list=feature_list
    )