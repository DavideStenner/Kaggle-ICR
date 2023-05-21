import numpy as np
import pandas as pd
from tqdm import tqdm

def normalize_data(
        data: pd.DataFrame, 
        new_min: float = 0., new_max: float = 1.
    )-> pd.DataFrame:
    
    min_, max_ = data.min(), data.max()
    
    return ((data - min_) / (max_ - min_)) * (new_max-new_min) + new_min

def binarize_data(data: pd.DataFrame) -> pd.DataFrame:
    threshold_ = data.median()
    return (data >= threshold_).astype(int)

def simulate_index(
        number_possible_sample: int, pretraining_step: int, 
        replace_sampling: bool 
    ) -> np.array:

    sim_index = np.random.choice(
        range(number_possible_sample),
        pretraining_step, replace=replace_sampling
    )
    return sim_index

def pipeline_tabaugmentation(
        data: pd.DataFrame, feature_list: list, pretraining_step: int, 
        original_tgt_label: str, inference: bool,
        replace_sampling: bool=True,
    ) -> pd.DataFrame:
    
    data['selected_target'] = -1
    data['target_as_feature'] = np.nan
    
    if inference:
        return data
    else:
        sim_data = data.copy()
        augmented_data = tabaugment(
            data=sim_data, feature_list=feature_list, 
            pretraining_step=pretraining_step, 
            original_tgt_label=original_tgt_label, replace_sampling=replace_sampling
        )
        append_original = pd.concat([data, augmented_data], ignore_index=True)
        return append_original

def tabaugment(
        data: pd.DataFrame, feature_list: list, pretraining_step: int, 
        original_tgt_label: str, replace_sampling: bool,
    ) -> pd.DataFrame:

    print(f'Applying tabaugmentation')

    data['selected_target'] = -1
    data['target_as_feature'] = np.nan

    #tabaugment only not missing values
    not_null_pos = np.where(data[feature_list].notna())
    row_index_not_na, col_index_not_na = not_null_pos

    #random extraction
    number_possible_sample = len(row_index_not_na)
    if (not replace_sampling) & (pretraining_step > number_possible_sample):
        print(f'Replacing pretraining_step with maximum number allowed {number_possible_sample}')
        pretraining_step = number_possible_sample

    simulated_index = simulate_index(
        number_possible_sample=number_possible_sample,
        pretraining_step=pretraining_step,
        replace_sampling=replace_sampling
    )
    
    simulated_row, simulated_target = (
        row_index_not_na[simulated_index], 
        col_index_not_na[simulated_index]
    )
    simulated_df = data.loc[simulated_row].reset_index(drop=True)
    number_simulation = simulated_df.shape[0]

    print('Calculating rescaled feature')
    rescaled_simulated_df = simulated_df.copy()

    for col in feature_list:
        rescaled_simulated_df[col] = binarize_data(rescaled_simulated_df[col])

    print('Adding augmentation')
    for i in tqdm(range(number_simulation), total=number_simulation):

        #get index of feature
        selected_target = feature_list[simulated_target[i]]
        #rescaled
        real_value = rescaled_simulated_df.loc[i, selected_target]
        #original target. add as feature
        target_value = simulated_df.loc[i, original_tgt_label]

        simulated_df.loc[
            i, 
            [
                'selected_target', selected_target, 
                'target_as_feature', original_tgt_label
            ]
        ] = (
            simulated_target[i],
            np.nan,
            target_value,
            real_value
        )

    return simulated_df