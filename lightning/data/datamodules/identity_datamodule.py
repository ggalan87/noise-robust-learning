from typing import Any, Callable, Union, Optional, List
import psutil

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.datamodules.vision_datamodule import LightningDataModule


class IdentityDataModule(LightningDataModule):
    """An alternative VisionDataModule for identity based datasets, e.g. person re-identification"""

    EXTRA_ARGS: dict = {}
    name: str = ""
    #: Dataset class to use
    dataset_cls: type
    #: A tuple describing the shape of the data
    dims: tuple

    def __init__(self,
                 data_dir: Optional[str] = None,
                 val_split: Union[int, float] = 0.2,
                 num_workers: int = 0,
                 normalize: bool = False,
                 batch_size: int = 32,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 train_transforms: Optional[Callable] = None,
                 val_transforms: Optional[Callable] = None,
                 test_transforms: Optional[Callable] = None,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms

        self.dataset_train = None
        self.dataset_gallery = None
        self.dataset_query = None

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """ Normally this contains dummy calls to dataset class in order to download the data. We leave it empty. """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset.

        Currently, val is not implemented, test for identity datasets consists of two parts, gallery and query
        """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self._train_transforms is None else self._train_transforms

            self.dataset_train = self.dataset_cls(
                self.data_dir, part_name='train', transform=train_transforms, relabel=True, **self.EXTRA_ARGS)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self._test_transforms is None else self._test_transforms

            self.dataset_gallery = self.dataset_cls(
                self.data_dir, part_name='gallery', transform=test_transforms, relabel=False, **self.EXTRA_ARGS)
            self.dataset_query = self.dataset_cls(
                self.data_dir, part_name='query', transform=test_transforms, relabel=False, **self.EXTRA_ARGS)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader. Returns gallery and query data loaders"""
        return [self._data_loader(self.dataset_gallery), self._data_loader(self.dataset_query)]

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError

    def _data_loader(self, dataset: Dataset, shuffle: bool = False, sampler: Sampler = None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=sampler
        )

    def default_transforms(self) -> Callable:
        """
        Omitted check for normalization / prefer to always normalize

        @return: a transforms object
        """
        default_transforms = transforms.Compose([transforms.ToTensor(), imagenet_normalization()])
        return default_transforms

