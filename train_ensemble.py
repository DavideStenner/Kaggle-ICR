import json

from script.utils import set_seed_globally
from script.contrastive.ensemble import get_retrieval_score

if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)

    with open('experiment_config_lgb.json') as config_experiment_file:
        config_experiment_lgb = json.load(config_experiment_file)

    with open('experiment_config_xgb.json') as config_experiment_file:
        config_experiment_xgb = json.load(config_experiment_file)

    set_seed_globally(config_project['RANDOM_STATE'])

    config_project['NAME_LGB'] = config_experiment_lgb['NAME']
    config_project['NAME_XGB'] = config_experiment_xgb['NAME']
    feature_list = config_project['ORIGINAL_FEATURE']

    print('starting to evaluate ensemble')
    get_retrieval_score(config_experiment=config_project, feature_list=feature_list, target_col=config_project['TARGET_COL'])