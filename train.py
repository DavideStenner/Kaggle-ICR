import json

from script.model import run_tabular_experiment, evaluate_experiment_score

if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)
    
    config_project.update(
        {
            'NAME': 'lgb',
            'AUGMENT': True,
            'NUM_AUGMENT': 5000,
            'LOG_EVALUATION': 250,
            'SAVE_MODEL': True,
            'SAVE_RESULTS_PATH': 'experiment'
        }
    )
    print('Starting Experiment', config_project['NAME'])

    PARAMS_LGB = {
        'tree_learner': 'voting',
        'boosting_type': 'gbdt',
        #this will be overwrite if augment
        'objective': 'binary',
        'n_jobs': -1,
        'num_leaves': 2**8,
        'learning_rate': 0.05,
        'feature_fraction': 0.75,
        'bagging_freq': 1,
        'bagging_fraction': 0.80,
        'lambda_l2': 1,
        'verbosity': -1,
        'n_round': 2000,
    }
    feature_list = config_project['ORIGINAL_FEATURE']
    
    run_tabular_experiment(
        config_experiment=config_project, params_lgb=PARAMS_LGB, 
        feature_list=feature_list, augment=config_project['AUGMENT'],
        pretraining_step=config_project['NUM_AUGMENT'],
        target_col=config_project['TARGET_COL'], 
    )
    if config_project['SAVE_MODEL']:
        evaluate_experiment_score(
            config_experiment=config_project, params_lgb=PARAMS_LGB, 
            feature_list=feature_list, augment=config_project['AUGMENT']
        )