import pickle
import torch
import csv
from pathlib import Path


# Implementation
def generate(dataset_name, noise_type, noise_rate, cli_root_path):
    results_info_path = cli_root_path / 'results_info' / dataset_name
    logs_path = cli_root_path / f'./lightning_logs/{dataset_name}_LitSolider'
    filename = f'{noise_type}_{noise_rate}.csv'

    with open(results_info_path / filename) as csvfile:
        reader = csv.reader(filter(lambda row: row[0] != "#", csvfile))
        row = next(reader)

        for v in row:
            if len(v.strip()) == 0:
                print('n/a')
                continue
            acc_path = logs_path / f'version_{v}/acc.pkl'
            acc = pickle.load(open(acc_path, 'rb'))
            few_acc = len(acc) < 39
            warning = ' (too few epochs)' if few_acc else ''
            acc = torch.tensor(acc)
            max_acc, last_acc = acc.max(), acc[-1]
            print('{:.2f}{}'.format(100 * max_acc.item(), warning))
            # print('{:.2f} ({:.2f}){}'.format(100 * max_acc.item(), 100 * last_acc.item(), warning))

# Options
dataset_name = 'market1501'

noise_configs = \
    {
        # 'ccn': ['0.1', '0.2', '0.5'],
        # 'idn': ['0.1', '0.2', '0.5'],
        'scn': ['0.25', '0.5']
    }

# noise_type = 'idn'
# noise_rate = '0.2'
cli_root_path = Path('/home/amidemo/devel/workspace/object_classifier_deploy/lightning/cli_pipelines')

for noise_type, noise_rates in noise_configs.items():
    for noise_rate in noise_rates:
        print('*' * 20)
        print(noise_type, noise_rate)
        generate(dataset_name, noise_type, noise_rate, cli_root_path)
