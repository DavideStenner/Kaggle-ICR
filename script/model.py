import os
import gc
import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

from typing import Tuple, Callable
from functools import partial

from script.loss import lgb_metric, lgb_augment_metric, table_augmentation_logloss
from script.augment import pipeline_tabaugmentation

AUGMENT_FEATURE = ['selected_target'] #'target_as_feature'

def get_dataset(
        data: pd.DataFrame, fold_: int, inference: bool,
        target_col: str, feature_list: list,
    ) -> lgb.Dataset:
    mask_fold = data['fold']==fold_ if inference else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)

    train_x = data[feature_list].to_numpy('float32')
    train_y = data[target_col].to_numpy('float32')

    lgb_dataset = lgb.Dataset(train_x, train_y)
    return lgb_dataset

def get_augment_dataset(
        data: pd.DataFrame, fold_: int, inference: bool,
        target_col: str, feature_list: list, pretraining_step: int,
    ) -> lgb.Dataset | Tuple[lgb.Dataset, Callable]:
    
    mask_fold = data['fold']==fold_ if inference else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)
    #tabaugment
    data = pipeline_tabaugmentation(
        data=data, feature_list=feature_list, 
        pretraining_step=pretraining_step, original_tgt_label=target_col, 
        inference=inference
    )

    train_feat_list = feature_list + AUGMENT_FEATURE
        
    train_x = data[train_feat_list].to_numpy('float32')
    train_y = data[target_col].to_numpy('float32')

    lgb_dataset = lgb.Dataset(train_x, train_y)

    if inference:
        return lgb_dataset
    else:
        #original target value
        mask_value_for_loss=(data['selected_target']==-1).values
        pretraining_loss = partial(table_augmentation_logloss, mask_value_for_loss)
        
        return lgb_dataset, pretraining_loss

def run_tabular_experiment(
        config_experiment: dict, params_lgb: dict,
        feature_list: list, augment: bool, pretraining_step: int,
        target_col: str = 'Class',
    ) -> None:

    if ('objective' not in params_lgb.keys()) & (not augment):
        raise KeyError('Need to set objective param for lgb if augment is set.')
    
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

    #set for normal experiment. if augment overwrite
    feval=(lgb_augment_metric if augment else lgb_metric)
    fobj=None

    for fold_ in range(config_experiment['N_FOLD']):
        print(f'\n\nStarting fold {fold_}\n\n\n')
        
        if augment:
            train_matrix, pretraining_loss = get_augment_dataset(
                data=train, fold_=fold_, inference=False, 
                target_col=target_col, feature_list=feature_list,
                pretraining_step=pretraining_step,
            )
            test_matrix = get_augment_dataset(
                data=train, fold_=fold_, 
                inference=True, target_col=target_col, 
                feature_list=feature_list,
                pretraining_step=pretraining_step,
            )

        if augment:
            fobj=pretraining_loss

        else:
            train_matrix = get_dataset(
                data=train, fold_=fold_, inference=False,
                target_col=target_col, feature_list=feature_list
            )

            test_matrix = get_dataset(
                data=train, fold_=fold_, inference=True,
                target_col=target_col, feature_list=feature_list
            )

        progress = {}

        callbacks_list = [
            lgb.record_evaluation(progress),
            lgb.log_evaluation(period=config_experiment['LOG_EVALUATION'], show_stdv=False)
        ]

        model = lgb.train(
            params=params_lgb,
            train_set=train_matrix, 
            num_boost_round=params_lgb['n_round'],
            valid_sets=[test_matrix],
            valid_names=['valid'],
            callbacks=callbacks_list,
            fobj=fobj,
            feval=feval
        )

        if config_experiment['SAVE_MODEL']:
            model.save_model(
                os.path.join(
                       save_path,
                    f'lgb_{fold_}.txt'
                )
            )

        model_list.append(model)
        progress_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()
        if config_experiment['SAVE_MODEL']:
            save_lgb_model(
                model_list=model_list, progress_list=progress_list,
                save_path=save_path
            )

def save_lgb_model(
        model_list: list, progress_list: list, save_path: str
    )->None:
        with open(
            os.path.join(
                save_path,
                'model_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)

        with open(
            os.path.join(
                save_path,
                'progress_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)

def evaluate_experiment_score(
        config_experiment: dict, params_lgb: dict, feature_list: list,
        augment: bool
    ) -> None:
    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'], 
        config_experiment['NAME']
    )
    
    # Find best epoch
    with open(
        os.path.join(
            save_path,
            'progress_list_lgb.pkl'
        ), 'rb'
    ) as file:
        progress_list_lgb = pickle.load(file)

    with open(
        os.path.join(
            save_path,
            'model_list_lgb.pkl'
        ), 'rb'
    ) as file:
        model_list_lgb = pickle.load(file)

        
    progress_dict_lgb = {
        'time': range(params_lgb['n_round']),
    }

    progress_dict_lgb.update(
            {
                f'loss_fold_{i}': progress_list_lgb[i]['valid']['balanced_log_loss']
                for i in range(config_experiment['N_FOLD'])
            }
        )

    progress_df_lgb = pd.DataFrame(progress_dict_lgb)

    progress_df_lgb['average_loss'] = progress_df_lgb.loc[
        :, ['loss' in x for x in progress_df_lgb.columns]
    ].mean(axis =1)
    
    progress_df_lgb['std_loss'] = progress_df_lgb.loc[
        :, ['loss' in x for x in progress_df_lgb.columns]
    ].std(axis =1)

    best_epoch_lgb = int(progress_df_lgb['average_loss'].argmin())
    best_score_lgb = progress_df_lgb['average_loss'].min()
    lgb_std = progress_df_lgb.loc[best_epoch_lgb, 'std_loss']

    print(f'Best epoch: {best_epoch_lgb}, CV-Loss: {best_score_lgb:.5f} ± {lgb_std:.5f}')

    best_result_lgb = {
        'best_epoch': best_epoch_lgb+1,
        'best_score': best_score_lgb
    }

    with open(
        os.path.join(
            save_path,
            'best_result_lgb.txt'
        ), 'w'
    ) as file:
        json.dump(best_result_lgb, file)

    explain_model(config_experiment, best_result_lgb, model_list_lgb, feature_list, augment)

def explain_model(
        config_experiment: dict, best_result_lgb: dict, 
        model_list_lgb: Tuple[lgb.Booster, ...], feature_list: list,
        augment: bool
    ) -> None:
    
    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'], 
        config_experiment['NAME']
    )
    
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = (
        feature_list + AUGMENT_FEATURE if augment
        else feature_list
    )

    for fold_, model in enumerate(model_list_lgb):
        feature_importances[f'fold_{fold_}'] = model.feature_importance(
            importance_type='gain', iteration=best_result_lgb['best_epoch']
        )

    feature_importances['average'] = feature_importances[
        [f'fold_{fold_}' for fold_ in range(config_experiment['N_FOLD'])]
    ].mean(axis=1)

    fig = plt.figure(figsize=(12,8))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
    plt.title(f"50 TOP feature importance over {config_experiment['N_FOLD']} average")

    fig.savefig(
        os.path.join(save_path, 'importance_plot.png')
    )