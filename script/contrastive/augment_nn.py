import numpy as np
import pandas as pd

from typing import Tuple
from random import sample
from itertools import combinations, chain
from operator import itemgetter 
from sklearn.model_selection import StratifiedKFold

def find_example(
        data: pd.DataFrame, row: pd.Series, index_: int, num_sample: int, 
        equal: bool, original_tgt_label:str
    ) -> list:
    target_sample = row[original_tgt_label] if equal else (1-row[original_tgt_label])

    similar_row = data.loc[
        (data.index != index_) &
        (data[original_tgt_label] == target_sample)
    ].sample(num_sample).index
    
    return list(similar_row)

def get_smart_combination(
        data: pd.DataFrame, original_tgt_label: str,
        number_equal: dict = {0: 4, 1: 16},
        number_unequal: int = 4
    ) -> Tuple[list, list]:

    current_index = []
    sampled_index = []

    num_0_0_equal = 0
    num_1_1_equal = 0
    num_sim_unequal = 0

    for index_, row in data.iterrows():
        curr_target = row[original_tgt_label]

        sampled_equal_index = find_example(
            data=data, row=row, index_=index_, num_sample=number_equal[curr_target], 
            equal=True, original_tgt_label=original_tgt_label
        )
        sampled_unequal_index = find_example(
            data=data, row=row, index_=index_, num_sample=number_unequal, 
            equal=False, original_tgt_label=original_tgt_label
        )

        sampled_index += sampled_equal_index
        sampled_index += sampled_unequal_index

        sampled_length = len(sampled_equal_index) + len(sampled_unequal_index)
        current_index += [index_] * sampled_length

        if curr_target == 0:
            num_0_0_equal += len(sampled_equal_index)
        else:
            num_1_1_equal += len(sampled_equal_index)

        num_sim_unequal += len(sampled_unequal_index)

    print(f'{num_sim_unequal} unequal; {num_1_1_equal} 1-1; {num_0_0_equal} 0-0')

    return current_index, sampled_index

def get_all_combination_stratified(
        data: pd.DataFrame, original_tgt_label: str,
        batch_size: int,
    ) -> Tuple[list, list]:
    
    print('Getting index')
    index_number = list(range(data.shape[0]))
    list_all_combination = list(
        combinations(
            index_number,
            2
        )
    )
    #shuffle index 
    list_all_combination = [
        sample(x, 2) for x in list_all_combination
    ]
    
    c_1_simulated = [c_1 for c_1, _ in list_all_combination]
    c_2_simulated = [c_2 for _, c_2 in list_all_combination]

    observation_1 = data.loc[c_1_simulated].reset_index()
    observation_2 = data.loc[c_2_simulated].reset_index()

    mask_unequal = (observation_1[original_tgt_label] + observation_2[original_tgt_label]) == 1

    mask_0_0 = (observation_1[original_tgt_label] + observation_2[original_tgt_label]) == 0
    mask_1_1 = (observation_1[original_tgt_label] + observation_2[original_tgt_label]) == 2

    sum_unequal, sum_0_0, sum_1_1 = sum(mask_unequal), sum(mask_0_0), sum(mask_1_1)
    equal_agg_num = min(sum_0_0, sum_1_1)

    num_sim_equal = min(sum_unequal, equal_agg_num)

    num_0_0_equal = sum_0_0#min(sum_0_0, num_sim_equal)
    num_1_1_equal = sum_1_1#min(sum_1_1, num_sim_equal)
    num_sim_unequal = min(sum_unequal, num_0_0_equal + num_1_1_equal)

    equal_0_0_index = sample(range(sum_0_0), num_0_0_equal)
    equal_1_1_index = sample(range(sum_1_1), num_1_1_equal)

    unequal_index = sample(range(sum_unequal), num_sim_unequal)

    unequal_sampler, equal_0_0_sampler, equal_1_1_sampler = (
        itemgetter(*unequal_index), itemgetter(*equal_0_0_index),
        itemgetter(*equal_1_1_index)
    )

    c_1_unequal_simulated = unequal_sampler(
        list(observation_1.loc[mask_unequal, 'index'])
    )
    
    c_2_unequal_simulated = unequal_sampler(
        list(observation_2.loc[mask_unequal, 'index'])
    )

    c_1_equal_0_0_simulated = equal_0_0_sampler(
        list(observation_1.loc[mask_0_0, 'index'])
    )
    c_2_equal_0_0_simulated = equal_0_0_sampler(
        list(observation_2.loc[mask_0_0, 'index'])
    )

    c_1_equal_1_1_simulated = equal_1_1_sampler(
        list(observation_1.loc[mask_1_1, 'index'])
    )
    c_2_equal_1_1_simulated = equal_1_1_sampler(
        list(observation_2.loc[mask_1_1, 'index'])
    )

    print(f'{num_sim_unequal} unequal; {num_1_1_equal} 1-1; {num_0_0_equal} 0-0')
    c_1_sampled, c_2_sampled = shuffle_simulation(
        simulation_list_1=[
            c_1_unequal_simulated, c_1_equal_0_0_simulated, c_1_equal_1_1_simulated
        ],
        simulation_list_2=[
            c_2_unequal_simulated, c_2_equal_0_0_simulated, c_2_equal_1_1_simulated
        ],
        batch_size=batch_size
    )
    return c_1_sampled, c_2_sampled

def random_simulation(simulation_list_1: list, simulation_list_2: list):
    c_1_simulated = np.array(list(chain(*simulation_list_1)))
    c_2_simulated = np.array(list(chain(*simulation_list_2)))

    number_index_simulated = len(c_1_simulated)
    sampled_index = sample(range(number_index_simulated), number_index_simulated)

    sampler_operator = itemgetter(*sampled_index)

    c_1_sampled, c_2_sampled = sampler_operator(c_1_simulated), sampler_operator(c_2_simulated)
    return c_1_sampled, c_2_sampled

def shuffle_simulation(simulation_list_1: list, simulation_list_2: list, batch_size: int):
    """
    ensure each batch has the same ratio of observation (shuffled) of each vector inside simulation list
    which are: unequal, equal 0_0, equal 1_1
    """
    simulation_label = [
        [i] * len(x)
        for i, x in enumerate(simulation_list_1)
    ]

    label_flatted = np.array(list(chain(*simulation_label)))
    simulation_flatted_1 = np.array(list(chain(*simulation_list_1)))
    simulation_flatted_2 = np.array(list(chain(*simulation_list_2)))

    n_batches = len(label_flatted) // batch_size

    shuffler = StratifiedKFold(n_splits=n_batches, shuffle=True, random_state=548652)

    pos_array = []

    for _, test_idx in shuffler.split(simulation_flatted_1, label_flatted):
        pos_array += list(test_idx)

    assert np.unique(pos_array).shape[0] == simulation_flatted_1.shape[0]

    simulation_shuffled_1 = simulation_flatted_1[pos_array].tolist()
    simulation_shuffled_2 = simulation_flatted_2[pos_array].tolist()

    return simulation_shuffled_1, simulation_shuffled_2