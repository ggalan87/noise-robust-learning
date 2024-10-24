import copy
import pickle
from typing import Any, Optional, Callable

from pathlib import Path
import re

import torch
from lightning.data.dataset_utils import pil_loader, disturb_records_targets
from lightning.data.datasets.identity_dataset import IdentityImageDataset
import pandas as pd
import numpy as np


class MSMT17DatasetPart(IdentityImageDataset):
    parts_folders = \
        {
            'train': 'train',
            'val': 'train',
            'gallery': 'test',
            'query': 'test'
        }

    def __init__(
            self,
            root: str,
            part_name: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            relabel=False,
            dataset_version=1,
            combine_all=False,
            training_variant=None,
            ignore_noisy_samples: bool = False
    ):
        super(MSMT17DatasetPart, self).__init__(root,
                                                part_name,
                                                transform=transform,
                                                target_transform=target_transform,
                                                relabel=relabel)

        variants = \
            [
                'symmetric_noise_0.1',
                'symmetric_noise_0.2',
                'symmetric_noise_0.5',
                'small_cluster_noise_0.25',
                'small_cluster_noise_0.5',
                'instance_dependent_noise_0.1',
                'instance_dependent_noise_0.2',
                'instance_dependent_noise_0.5'
            ]

        if training_variant is not None and training_variant not in variants:
            raise AssertionError('Unsupported training variant!')

        if training_variant is not None and combine_all is True:
            raise NotImplementedError('Noise was generated without the combined dataset.')

        self.dataset_version = dataset_version
        self.training_variant = training_variant
        self.ignore_noisy_samples = ignore_noisy_samples
        self.combine_all = combine_all

        self._load_data()

    def _load_data(self):
        # Modified from:
        # https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/image/msmt17.py

        if self.part_name == 'train' and self.training_variant is not None:
            noisy_info_file = self.raw_folder.parent / 'noisy_labels' / f'{self.training_variant}.pkl'
            with open(noisy_info_file, 'rb') as f:
                noisy_info = pickle.load(f)
        else:
            noisy_info = None

        part_file = self.raw_folder / f'list_{self.part_name}.txt'

        with open(part_file, 'r') as f:
            image_infos = f.readlines()

        if self.part_name == 'train' and self.combine_all:
            val_file = self.raw_folder / f'list_val.txt'
            with open(val_file, 'r') as f:
                image_infos.extend(f.readlines())

        images_folder = self.get_part_directory()

        if noisy_info is not None:
            unique_targets = torch.unique(noisy_info['targets'])
            if 'small_cluster' in self.training_variant:
                print('Reduced labels set. Need extra relabel of the pids.')
                noisy_info['orig_to_new_labels'] = {int(orig_label): new_label for new_label, orig_label in enumerate(unique_targets)}
            else:
                noisy_info['orig_to_new_labels'] = None

        for i, entry in enumerate(image_infos):
            img_relative_path, pid = entry[:-1].split(' ')  # eliminate the newline and split into [path, pid]
            pid = int(pid)
            camid = int(img_relative_path.split('_')[2]) - 1  # index starts from 0
            img_path = images_folder / img_relative_path

            data_entry = \
                {
                    'image_path': str(img_path),
                    'target': pid,
                    'camera_id': camid
                }

            if noisy_info is not None:
                data_entry['target_orig'] = data_entry['target']

                # Check for possible extra relabel map
                relabel_map = noisy_info['orig_to_new_labels']
                data_entry['target'] = noisy_info['targets'][i] if relabel_map is None \
                    else relabel_map[int(noisy_info['targets'][i])]

                data_entry['is_noisy'] = noisy_info['noisy_indices'][i]

            self.data.append(data_entry)

    @property
    def raw_folder(self) -> Path:
        return Path(self.root) / f'msmt17' / f'MSMT17_V{self.dataset_version}'


def pass_dataset():
    msmt17_train = MSMT17DatasetPart('/data/datasets/', part_name='train', relabel=True,
                                     training_variant='small_cluster_noise_0.25')

    noisy = 0
    non_noisy = 0
    for entry in msmt17_train.data:
        if entry['is_noisy']:
            noisy += 1
        else:
            non_noisy += 1

    print(noisy, non_noisy)


def generate_noisy_labels():
    def save_noisy_data(data, output_path):
        data_df = pd.DataFrame(data)
        targets = torch.tensor(data_df['target'].values)
        noisy_indices = torch.tensor(data_df['is_noisy'].values)
        noisy_data = \
            {
                'targets': targets,
                'noisy_indices': noisy_indices
            }

        with open(output_path, 'wb') as f:
            pickle.dump(noisy_data, f)

    from lightning_lite.utilities.seed import seed_everything
    # seed everything
    seed_everything(13)

    msmt17_train = MSMT17DatasetPart('/data/datasets/', part_name='train', relabel=True, training_variant=None)

    msmt17_train_0_5_symmetric = (
        disturb_records_targets(msmt17_train.data, labels_noise_perc={'symmetric': 0.5}, key_to_disturb='target'))
    msmt17_train_0_2_symmetric = (
        disturb_records_targets(msmt17_train.data, labels_noise_perc={'symmetric': 0.2}, key_to_disturb='target'))
    msmt17_train_0_1_symmetric = (
        disturb_records_targets(msmt17_train.data, labels_noise_perc={'symmetric': 0.1}, key_to_disturb='target'))

    noisy_labels_root = Path('/media/amidemo/Data/object_classifier_data/datasets/msmt17/noisy_labels/')
    save_noisy_data(msmt17_train_0_1_symmetric, noisy_labels_root / 'symmetric_noise_0.1.pkl')
    save_noisy_data(msmt17_train_0_2_symmetric, noisy_labels_root / 'symmetric_noise_0.2.pkl')
    save_noisy_data(msmt17_train_0_5_symmetric, noisy_labels_root / 'symmetric_noise_0.5.pkl')


if __name__ == '__main__':
    generate_noisy_labels()
    # pass_dataset()
