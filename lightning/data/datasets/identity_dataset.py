from typing import Any, Optional, Callable, Dict
from torchvision.datasets import VisionDataset
from pathlib import Path
import copy
from lightning.data.dataset_utils import pil_loader

class IdentityImageDataset(VisionDataset):
    """
    A VisionDataset subclass which aims to serve as base class of IdenityDatasets, e.g. datasets for re-identification
    which usually have already splits for train, gallery, query parts


    https://github.com/pytorch/vision/issues/215
    https://github.com/pytorch/vision/issues/230
    https://github.com/pytorch/vision/issues/5324
    """

    # A dict which maps dataset part names to paths relative to dataset root, e.g. 'train': <dataset_root>/train_dir
    parts_folders: Dict

    def __init__(
            self,
            root: str,
            part_name: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            relabel=False
    ) -> None:
        super(IdentityImageDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.part_name = part_name
        self.relabel = relabel
        self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        # Get a copy of the entry, such that the original data are not affected below / need to keep only shallow info
        # and not the actual image data. For now, we do shallow copy because we expect that none of the data are nested.
        data_entry = copy.copy(self.data[index])

        image_path = data_entry['image_path']

        image = pil_loader(image_path)

        if self.transform is not None:
            # TODO: decide how to pass the target (label) for possible "relabel transform"
            image = self.transform(image)

        data_entry['image'] = image
        data_entry['data_idx'] = index

        return data_entry

    def get_part_directory(self) -> Path:
        if self.part_name not in self.parts_folders:
            raise ValueError(f'Folder name for part {self.part_name} is not provided.')

        part_directory = self.raw_folder / self.parts_folders[self.part_name]

        if not part_directory.exists():
            raise ValueError(f'Path for part {self.part_name} does not exist: {part_directory}')

        return part_directory

    @property
    def name(self):
        return self.part_name

    @property
    def raw_folder(self) -> Path:
        raise NotImplementedError
