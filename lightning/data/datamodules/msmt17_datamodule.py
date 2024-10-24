from typing import Any, Optional, Union, List
from dataclasses import dataclass, field, asdict

from torch.utils.data import DataLoader
from torchvision import transforms
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from lightning.data.datamodules import IdentityDataModule
from lightning.data.samplers import RandomIdentitySampler
from lightning.data.default_transforms import SoliderTransforms, SoliderTrainTransforms
from lightning.data.datasets import MSMT17DatasetPart

from lightning.data.data_modules import patch_visiondatamodule


@dataclass
class MSMT17Args:
    training_variant: Optional[str] = None
    ignore_noisy_samples: bool = False
    dataset_version: int = 1
    combine_all: bool = False


class MSMT17DataModule(IdentityDataModule):
    name: str = "msmt17"
    dataset_cls = MSMT17DatasetPart
    # dims = (3, 128, 64)

    def __init__(
            self,
            data_dir: Optional[str] = None,
            val_split: Union[int, float] = 0.2,
            num_workers: int = 0,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            dataset_args: MSMT17Args = None,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            data_dir,
            val_split,
            num_workers,
            normalize,
            batch_size,
            seed,
            shuffle,
            pin_memory,
            drop_last,
            *args,
            **kwargs,
        )

        if dataset_args:
            self.EXTRA_ARGS = asdict(dataset_args)
            if dataset_args.training_variant and 'small_cluster_noise_0.5' in dataset_args.training_variant:
                self.num_classes = 521
            elif dataset_args.training_variant and 'small_cluster_noise_0.25' in dataset_args.training_variant:
                self.num_classes = 781
            else:
                self.num_classes = 1041
        else:
            #
            self.num_classes = 1041

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # super method calls the dataset to ensure data are downloaded. We don't need it here
        pass

    def default_transforms(self):
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                normalize_transform,
            ]
        )
        return transform

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader. Override such that we return a list that contains gallery and query data loaders"""
        return [self._data_loader(self.dataset_gallery), self._data_loader(self.dataset_query)]


def pass_dataloader():
    patch_visiondatamodule(
        sampler_class=RandomIdentitySampler,
        batch_size=64,
        num_instances=8,
        id_key='target',
        fix_samples=True,
        num_epochs=35,
    )

    dm = MSMT17DataModule(data_dir="/media/amidemo/Data/object_classifier_data/datasets",
                              batch_size=64,
                              train_transforms=SoliderTrainTransforms(),
                              val_transforms=SoliderTransforms(),
                              test_transforms=SoliderTransforms(),
                              val_split=0.0,
                              dataset_args=MSMT17Args(training_variant='symmetric_noise_0.5'))
    dm.setup('fit')

    train_dataloader = dm.train_dataloader()

    sampler = train_dataloader.sampler
    print(sampler.n_iterations)

    dm.setup('test')

    gallery_dataloader, query_dataloader = dm.test_dataloader()
    pass


if __name__ == '__main__':
    pass_dataloader()
