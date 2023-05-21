import numpy as np
import pandas as pd

from typing import Tuple
from itertools import combinations

def get_all_combination(data: pd.DataFrame, inference: bool) -> Tuple[list, list]:
    index_number = list(range(data.shape[0]))
    list_all_combination = list(
        combinations(
            index_number,
            2
        )
    )
    #tabaugment only not missing values
    if num_simulation is None:
        num_simulation = len(list_all_combination)
        if not inference:
            print(f'Applying contrastive training')
            print(f'Using maximum number of combination: {num_simulation}')

    c_1_simulated = [c_1 for c_1, _ in list_all_combination[:num_simulation]]
    c_2_simulated = [c_2 for _, c_2 in list_all_combination[:num_simulation]]
    return c_1_simulated, c_2_simulated

def get_stratified_example(
        data: pd.DataFrame, original_tgt_label: str, 
        num_simulation: int = 20
    ) -> Tuple[list, list]:

    c_1_simulated = []
    c_2_simulated = []
    target_sample = {
        0: 4,
        1: 20
    }
    for row in range(data.shape[0]):
        curr_target = data.loc[row, original_tgt_label]
        
        equal_idxs = np.where(data[original_tgt_label] == curr_target)[0]
        negative_idxs = np.where(data[original_tgt_label] != curr_target)[0]

        sampled_equal_idxs = np.random.choice(equal_idxs, size=target_sample[curr_target], replace=False).tolist()
        sampled_negative_idxs = np.random.choice(negative_idxs, size=target_sample[1-curr_target], replace=False).tolist()
        sampled_idxs = sampled_equal_idxs + sampled_negative_idxs
        
        c_1_simulated += [row for _ in range(len(sampled_idxs))]
        c_2_simulated += sampled_idxs
        
    return c_1_simulated, c_2_simulated

def contrastive_pipeline(
        data: pd.DataFrame, feature_list: list, inference: bool,
        original_tgt_label: str, num_simulation: int = None
    ) -> pd.DataFrame:

    # c_1_simulated, c_2_simulated = get_all_combination(data, inference)
    c_1_simulated, c_2_simulated = get_stratified_example(data, original_tgt_label)

    col_used = feature_list + [original_tgt_label]

    c_1_data = data.loc[
        c_1_simulated, col_used
    ].reset_index(drop=True)

    c_2_data = data.loc[
        c_2_simulated, col_used
    ].reset_index(drop=True)

    dataset_contrast = pd.DataFrame(
        np.abs(
            c_1_data[feature_list].values - 
            c_2_data[feature_list].values
        ), columns=feature_list
    )
    dataset_contrast['target_contrast'] = (
        c_2_data[original_tgt_label] == c_1_data[original_tgt_label]
    ).astype(int)

    return dataset_contrast