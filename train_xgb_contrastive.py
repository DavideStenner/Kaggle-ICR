import json

from script.utils import set_seed_globally
from script.contrastive.xgb_model import run_contrastive_xgb_experiment, evaluate_contrastive_xgb_score

if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)
    
    set_seed_globally(config_project['RANDOM_STATE'])

    with open('experiment_config_xgb.json') as config_experiment_file:
        config_experiment = json.load(config_experiment_file)
    
    assert set(
            [
                'NAME', 'NUM_SIMULATION', 'LOG_EVALUATION', 
                'SAVE_MODEL', 'INCREASE', 'TRAIN_MODEL', 'SAVE_RESULTS_PATH'
            ]
        ).issubset(set(config_experiment.keys()))

    config_project.update(config_experiment)

    print('Starting xgb Experiment', config_project['NAME'])

    PARAMS_XGB = {
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'reg_lambda': 1,
        'learning_rate':0.01,
        'subsample':0.75,
        'colsample_bytree':0.8,
        'nthread' : -1,
        'n_round': 1000,
        'seed': config_project['RANDOM_STATE']
    }

    feature_list = config_project['ORIGINAL_FEATURE']
    
    if config_project['TRAIN_MODEL']:
        run_contrastive_xgb_experiment(
            config_experiment=config_project, params_model=PARAMS_XGB, 
            feature_list=feature_list,
            num_simulation=config_project['NUM_SIMULATION'],
            target_col=config_project['TARGET_COL'], 
        )
    if config_project['SAVE_MODEL']:
        evaluate_contrastive_xgb_score(
            config_experiment=config_project, params_model=PARAMS_XGB, 
            feature_list=feature_list,
            target_col=config_project['TARGET_COL']
        )