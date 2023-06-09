
import json
import torch

from script.utils import set_seed_globally
from script.contrastive.nn_model import run_nn_contrastive_experiment

if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)

    set_seed_globally(config_project['RANDOM_STATE'])
    
    with open('experiment_config_contrastive_nn.json') as config_experiment_file:
        config_experiment = json.load(config_experiment_file)
    
    assert set(
            [
                'NAME', 'LOG_EVALUATION', 
                'SAVE_MODEL', 'INCREASE', 'TRAIN_MODEL', 'SAVE_RESULTS_PATH'
            ]
        ).issubset(set(config_experiment.keys()))

    config_project.update(config_experiment)

    print('Starting lgb Experiment', config_project['NAME'])

    config_model = {
        'batch_size': 16,
        'num_workers': 4,
        #huggingface model
        #entire script debug run
        'debug_run': False,
        #enable dev run on py light
        'dev_run': False,
        'n_fold': 5,
        'random_state': config_project['RANDOM_STATE'],
        'max_epochs_pretraining': 1,
        'max_epochs': 15,
        #number of step. disable with -1.
        'max_steps': -1,
        #trainer parameter --> check loss every n step. put 0.95 to disable this.
        'val_check_interval': 0.95,
        'accelerator': "gpu" if torch.cuda.is_available() else "cpu",
        'lr': 1e-4,
        #used for logging
        'version_experiment': 'deberta_benchmark',
        'progress_bar': False,
        'num_features': 56
    }
    feature_list = config_project['ORIGINAL_FEATURE']
    
    if config_project['TRAIN_MODEL']:
        run_nn_contrastive_experiment(
            config_experiment=config_project, config_model=config_model, 
            feature_list=feature_list,
            target_col=config_project['TARGET_COL'], 
        )