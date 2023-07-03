import os
import json
import glob

import pandas as pd
import pytorch_lightning as pl

from script.contrastive.nn_model import ContrastiveClassifier
from script.loss import calc_log_loss_weight
from script.contrastive.nn_dataset import get_training_dataset_loader

def define_folder_structure(config_experiment: dict, fold_: int):
    log_folder = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'],
        config_experiment['NAME'],
        'log',
        f'log_fold_{fold_}'
    )
    plot_folder = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'],
        config_experiment['NAME'],
        'plot',
        f'plot_fold_{fold_}'
    )
    model_folder = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'],
        config_experiment['NAME'],
        'model',
        f'model_fold_{fold_}'
    )

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    else:
        print('Deleting previous image')
        image_path_list = os.listdir(plot_folder)

        for image_path in image_path_list:
            os.remove(os.path.join(plot_folder, image_path))

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    return log_folder, plot_folder, model_folder

def run_nn_contrastive_experiment(
        config_experiment: dict, config_model: dict, 
        feature_list: list, target_col: str,
    ) -> None:
    
    train = pd.read_pickle(
        os.path.join(config_experiment['PATH_DATA'], 'processed_data.pkl')
    )[feature_list + ['fold', target_col, 'Alpha']]

    train[feature_list] = (train[feature_list]-train[feature_list].mean())/train[feature_list].std()
    train[feature_list] = train[feature_list].fillna(0)

    for fold_ in range(config_experiment['N_FOLD']):

        log_folder, plot_folder, model_folder = define_folder_structure(config_experiment, fold_)
        
        w_0, w_1 = calc_log_loss_weight(
            train.loc[
                train['fold']==fold_, config_experiment['TARGET_COL']
            ].values
        )
        config_model['pos_weight'] = w_1/w_0
        config_model['plot_folder'] = plot_folder

        print(f'\n\nStarting fold {fold_}\n\n\n')
        print('\n\nStarting Pretraining\n\n\n')
        (
            train_loader_contrastive, valid_loader_contrastive,
            train_loader_training, valid_loader_training
        ) = get_training_dataset_loader(
            config_model=config_model, train=train, 
            fold_=fold_, target_col=target_col, feature_list=feature_list,
            batch_size=config_model['batch_size_pretraining']
        )

        loggers_pretraining = pl.loggers.CSVLogger(
            save_dir=log_folder,
            name='pretraining',
            version=config_model['version_experiment']
        )

        contrastive_trainer = pl.Trainer(
            max_epochs=config_model['max_epochs_pretraining'],
            max_steps=config_model['max_steps'],
            fast_dev_run=config_model['dev_run'], 
            accelerator=config_model['accelerator'],
            val_check_interval=config_model['val_check_interval_pretraining'],
            enable_progress_bar=(config_model['debug_run']) | (config_model['progress_bar']),
            num_sanity_val_steps=config_model['num_sanity_val_steps_pretraining'],
            logger=[loggers_pretraining],
            gradient_clip_val=config_model['gradient_clip_val_pretraining'],
            accumulate_grad_batches=config_model['accumulate_grad_batches_pretraining'],
            check_val_every_n_epoch=config_model['check_val_every_n_epoch_pretraining'],
            enable_checkpointing=False
        )
        
        print_dataset = valid_loader_training if config_model['print_pretraining'] else None

        model_ = ContrastiveClassifier(config=config_model, valid_dataset=print_dataset)

        contrastive_trainer.fit(model_, train_loader_contrastive, valid_loader_contrastive)
        model_.init_classifier_setup()

        loggers_training = pl.loggers.CSVLogger(
            save_dir=log_folder,
            name='training',
            version=config_model['version_experiment']
        )

        classifier_trainer = pl.Trainer(
            max_epochs=config_model['max_epochs'],
            max_steps=config_model['max_steps'],
            fast_dev_run=config_model['dev_run'], 
            accelerator=config_model['accelerator'],
            val_check_interval=config_model['val_check_interval'],
            enable_progress_bar=(config_model['debug_run']) | (config_model['progress_bar']),
            num_sanity_val_steps=config_model['num_sanity_val_steps'],
            logger=[loggers_training],
            gradient_clip_val=config_model['gradient_clip_val'],
            accumulate_grad_batches=config_model['accumulate_grad_batches'],
            enable_checkpointing=False
        )
        print('\n\nStarting training\n\n')
        classifier_trainer.fit(model_, train_loader_training, valid_loader_training)

def eval_nn_contrastive_experiment(
    config_experiment: dict, step: str, 
) -> None:
    assert step in ['training', 'pretraining']
    loss_name = 'val_loss' if step=='pretraining' else 'val_comp_loss'

    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'],
        config_experiment['NAME']
    )
    path_results = os.path.join(
        save_path, 
        f'log\log_fold_*\{step}\*\metrics.csv'
    )
    metric_list = glob.glob(path_results)
    data = [
        pd.read_csv(path)
        for path in metric_list
    ]
    progress_dict = {
        'step': data[0]['step'],
    }

    progress_dict.update(
        {
            f"{loss_name}_fold_{i}": data[i][loss_name]
            for i in range(5)
        }
    )
    progress_df = pd.DataFrame(progress_dict)
    progress_df[f"average_{loss_name}"] = progress_df.loc[
        :, [loss_name in x for x in progress_df.columns]
    ].mean(axis =1)

    best_epoch = progress_df[f"average_{loss_name}"].argmin()
    best_step = progress_df.loc[
        best_epoch, "step"
    ]
    best_score = progress_df[f"average_{loss_name}"].min()

    best_score = {
        'best_epoch': int(best_epoch),
        'best_step': int(best_step),
        'best_score': best_score
    }
    print('\n')
    print(f'Best CV {loss_name} score for {step}')
    print(best_score)
    print('\n')
    
    with open(
            os.path.join(
                save_path,
                f'best_result_nn_{step}.txt'
            ), 'w'
        ) as file:
            json.dump(best_score, file)