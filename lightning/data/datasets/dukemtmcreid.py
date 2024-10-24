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


class DukeMTMCreIDDatasetPart(IdentityImageDataset):
    parts_folders = \
        {
            'train': 'bounding_box_train',
            'gallery': 'bounding_box_test',
            'query': 'query'
        }

    def __init__(
            self,
            root: str,
            part_name: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            relabel=False,
            training_variant=None,
            ignore_noisy_samples: bool = False
    ):
        super(DukeMTMCreIDDatasetPart, self).__init__(root,
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

        self.training_variant = training_variant
        self.ignore_noisy_samples = ignore_noisy_samples

        self._load_data()

    def _load_data(self):
        # Modified from:
        # https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/image/dukemtmcreid.py

        if self.part_name == 'train' and self.training_variant is not None:
            noisy_info_file = self.raw_folder.parent / 'noisy_labels' / f'{self.training_variant}.pkl'
            with open(noisy_info_file, 'rb') as f:
                noisy_info = pickle.load(f)
        else:
            noisy_info = None

        image_name_pattern = re.compile(r'([-\d]+)_c(\d)')

        images_folder = self.get_part_directory()

        assert images_folder.exists()

        img_paths = list(Path(images_folder).rglob('*.jpg'))

        person_ids = set()

        for img_path in img_paths:
            pid, _ = map(int, image_name_pattern.search(img_path.name).groups())
            if pid == -1:
                continue  # junk images are just ignored
            person_ids.add(pid)

        pid2label = {pid: label for label, pid in enumerate(person_ids)}

        if noisy_info is not None:
            unique_targets = torch.unique(noisy_info['targets'])
            if len(unique_targets) != len(person_ids):
                print('Reduced labels set. Need extra relabel of the pids.')
                noisy_info['orig_to_new_labels'] = {int(orig_label): new_label for new_label, orig_label in enumerate(unique_targets)}
            else:
                noisy_info['orig_to_new_labels'] = None

        for i, img_path in enumerate(img_paths):
            pid, camid = map(int, image_name_pattern.search(img_path.name).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if self.relabel:
                pid = pid2label[pid]

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

            if self.ignore_noisy_samples and data_entry['is_noisy']:
                continue

            self.data.append(data_entry)

        # Map pid to a corresponding name. This is the same with prefix, but keep it here for completeness /
        # compatibility with named datasets
        self.classes = {pid: f'ID_{pid}' for pid, _ in pid2label.items()}

    @property
    def raw_folder(self) -> Path:
        return Path(self.root) / f'dukemtmcreid' / 'DukeMTMC-reID'

def pass_dataset():
    dukemtmcreid_train = DukeMTMCreIDDatasetPart('/data/datasets/', part_name='train', relabel=True,
                                         training_variant='small_cluster_noise_0.25')

    noisy = 0
    non_noisy = 0
    for entry in dukemtmcreid_train.data:
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

    dukemtmcreid_train = DukeMTMCreIDDatasetPart('/data/datasets/', part_name='train', relabel=True, training_variant=None)

    dukemtmcreid_train_0_5_symmetric = (
        disturb_records_targets(dukemtmcreid_train.data, labels_noise_perc={'symmetric': 0.5}, key_to_disturb='target'))
    dukemtmcreid_train_0_2_symmetric = (
        disturb_records_targets(dukemtmcreid_train.data, labels_noise_perc={'symmetric': 0.2}, key_to_disturb='target'))
    dukemtmcreid_train_0_1_symmetric = (
        disturb_records_targets(dukemtmcreid_train.data, labels_noise_perc={'symmetric': 0.1}, key_to_disturb='target'))

    noisy_labels_root = Path('/media/amidemo/Data/object_classifier_data/datasets/dukemtmcreid/noisy_labels/')
    save_noisy_data(dukemtmcreid_train_0_1_symmetric, noisy_labels_root / 'symmetric_noise_0.1.pkl')
    save_noisy_data(dukemtmcreid_train_0_2_symmetric, noisy_labels_root / 'symmetric_noise_0.2.pkl')
    save_noisy_data(dukemtmcreid_train_0_5_symmetric, noisy_labels_root /'symmetric_noise_0.5.pkl')


if __name__ == '__main__':
    generate_noisy_labels()
    # pass_dataset()
