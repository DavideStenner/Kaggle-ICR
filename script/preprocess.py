import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

def preprocess_data(config: dict) -> None:

    greeks = pd.read_csv(
        os.path.join(
        config['PATH_DATA_ORIGINAL'],
        'greeks.csv'
        )
    )
    train = pd.read_csv(
        os.path.join(
        config['PATH_DATA_ORIGINAL'],
        'train.csv'
        )
    )
    greeks['Epsilon'] = pd.to_datetime(
        greeks['Epsilon'].replace('Unknown', np.nan)
    )

    fold_info = pd.merge(
        train[['Id', 'Class', 'BN', 'EJ']], 
        greeks, on='Id'
    )
    age_ = (np.round(fold_info['BN']/0.3531, 1).astype(int))
    age_binned = pd.cut(age_, config['N_FOLD'], labels=range(config['N_FOLD'])).astype(str)
    
    #condition + age + sex
    strat_col = (
        fold_info['Alpha'].astype(str) + '_' + 
        age_binned + '_' + 
        # fold_info['EJ'] + '_' + 
        # fold_info['Beta'] + '_' + 
        fold_info['Gamma']
    )

    split = StratifiedKFold(n_splits=config['N_FOLD'], shuffle=True, random_state=config['RANDOM_STATE'])
    iterator_split = enumerate(split.split(fold_info, strat_col))
    fold_info['fold'] = int(-1)

    for fold_, (_, test_index) in iterator_split:
        fold_info.loc[test_index, 'fold'] = int(fold_)

    assert (fold_info['fold'] == -1).sum() == 0
    assert fold_info['fold'].nunique() == config['N_FOLD']

    train = pd.merge(
        train,
        fold_info[
            [
                'Id', 'Alpha', 'Beta', 
                'Gamma', 'Delta', 'Epsilon',
                'fold'
            ]
        ], on='Id'
    )
    train['EJ'] = train['EJ'].map(
        {
            'A': 0,
            'B': 1
        }
    ).astype('uint8')

    train.to_pickle(
        os.path.join(
            config['PATH_DATA'],
            'processed_data.pkl'
        )
    )