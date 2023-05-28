
import json

from script.utils import set_seed_globally
from script.contrastive.lgb_model import run_lgb_experiment, evaluate_lgb_score

if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)

    set_seed_globally(config_project['RANDOM_STATE'])
    
    with open('experiment_config_lgb.json') as config_experiment_file:
        config_experiment = json.load(config_experiment_file)
    
    assert set(
            [
                'NAME', 'NUM_SIMULATION', 'LOG_EVALUATION', 
                'SAVE_MODEL', 'INCREASE', 'TRAIN_MODEL', 'SAVE_RESULTS_PATH'
            ]
        ).issubset(set(config_experiment.keys()))

    config_project.update(config_experiment)

    print('Starting lgb Experiment', config_project['NAME'])

    PARAMS_LGB = {
        'tree_learner': 'voting',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_jobs': -1,
        'num_leaves': 2**8,
        'learning_rate': 0.01,
        'feature_fraction': 0.75,
        'bagging_freq': 1,
        'bagging_fraction': 0.80,
        'lambda_l2': 1,
        'verbosity': -1,
        'n_round': 500,
        'seed': config_project['RANDOM_STATE']
    }
    feature_list = config_project['ORIGINAL_FEATURE']
    
    if config_project['TRAIN_MODEL']:
        run_lgb_experiment(
            config_experiment=config_project, params_model=PARAMS_LGB, 
            feature_list=feature_list,
            num_simulation=config_project['NUM_SIMULATION'],
            target_col=config_project['TARGET_COL'], 
        )
    if config_project['SAVE_MODEL']:
        evaluate_lgb_score(
            config_experiment=config_project, params_model=PARAMS_LGB, 
            feature_list=feature_list,
            target_col=config_project['TARGET_COL']
        )