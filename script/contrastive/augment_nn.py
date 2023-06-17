import pandas as pd

from typing import Tuple
from random import sample
from itertools import combinations
from operator import itemgetter 

def get_all_combination_stratified(
        data: pd.DataFrame, original_tgt_label: str
    ) -> Tuple[list, list]:
    
    print('Getting index')
    index_number = list(range(data.shape[0]))
    list_all_combination = list(
        combinations(
            index_number,
            2
        )
    )
    c_1_simulated = [c_1 for c_1, _ in list_all_combination]
    c_2_simulated = [c_2 for _, c_2 in list_all_combination]

    observation_1 = data.loc[c_1_simulated].reset_index()
    observation_2 = data.loc[c_2_simulated].reset_index()

    mask_unequal = (observation_1[original_tgt_label] + observation_2[original_tgt_label]) == 1

    mask_0_0 = (observation_1[original_tgt_label] + observation_2[original_tgt_label]) == 0
    mask_1_1 = (observation_1[original_tgt_label] + observation_2[original_tgt_label]) == 2

    sum_unequal, sum_0_0, sum_1_1 = sum(mask_unequal), sum(mask_0_0), sum(mask_1_1)
    equal_agg_num = min(sum_0_0, sum_1_1)

    num_sim_unequal = num_sim_equal = min(sum_unequal, equal_agg_num)
    num_0_0_equal = min(sum_0_0, num_sim_equal)
    num_1_1_equal = min(sum_1_1, num_sim_equal)
    
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
    c_1_simulated = c_1_unequal_simulated + c_1_equal_0_0_simulated + c_1_equal_1_1_simulated
    c_2_simulated = c_2_unequal_simulated + c_2_equal_0_0_simulated + c_2_equal_1_1_simulated

    number_index_simulated = len(c_1_simulated)
    sampled_index = sample(range(number_index_simulated), number_index_simulated)

    sampler_operator = itemgetter(*sampled_index)

    c_1_sampled, c_2_sampled = sampler_operator(c_1_simulated), sampler_operator(c_2_simulated)
    return c_1_sampled, c_2_sampled
