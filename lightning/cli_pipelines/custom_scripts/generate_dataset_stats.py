from dataclasses import dataclass
import torch
import pandas as pd
from lightning.data.datasets import Market1501DatasetPart, DukeMTMCreIDDatasetPart, MSMT17DatasetPart

dataset_part_names = ['train', 'gallery', 'query']
dataset_classes = {
    'market1501': Market1501DatasetPart,
    'dukemtmcreid': DukeMTMCreIDDatasetPart,
    'msmt17': MSMT17DatasetPart
}


@dataclass
class PartStats:
    part_name: str
    n_samples: int
    n_ids: int
    n_cameras: int


for dataset_name, dataset_class in dataset_classes.items():
    part_stats = []
    for part_name in dataset_part_names:
            needs_relabel = part_name == 'train'
            dataset = dataset_class(root='/data/datasets/', part_name=part_name, relabel=needs_relabel)
            data_df = pd.DataFrame(dataset.data)
            targets = torch.tensor(data_df['target'].values)
            cameras = torch.tensor(data_df['camera_id'].values)

            n_samples = len(targets)
            n_targets = len(torch.unique(targets))
            n_cameras = len(torch.unique(cameras))

            part_stats.append(PartStats(part_name=part_name, n_samples=n_samples, n_ids=n_targets, n_cameras=n_cameras))

    part_names = []
    table = []
    for part in part_stats:
        part_names.append(part.part_name)
        table.append([part.n_samples, part.n_ids, part.n_cameras])

    df = pd.DataFrame(table, columns=['#samples', '#ids', '#cameras'], index=part_names)
    print(f'\n{dataset_name}')
    print(df)
