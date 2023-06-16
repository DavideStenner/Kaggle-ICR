import subprocess
from script.nn_experiment import run_single_experiment

if __name__ == '__main__':

    for x in [1, 2, 3, 4, 5]:
        model_kwars = {
            'max_epochs_pretraining': x,
            'max_epochs': 20,
            'embedding_size': 2048,
        }
        print(f'\n\n\nStarting experiment with {x} pretraining step\n\n\n')
        run_single_experiment(experiment_position=f'nn_256/{x}', model_kwars=model_kwars)