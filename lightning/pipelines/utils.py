import sys
import inspect
import torch
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
import features_storage


def get_features(model, stage, data_module: LightningDataModule, batch_keys=('image', 'target')):
    """
    Obtains features and corresponding class labels given a dataset and a model at a given stage

    @param model: The model from which we will extract the features
    @param stage: The stage in pytorch lightning corresponds to training part that is fit (training stage) and test
    (testing stage)
    @param data_module: A LightningDataModule that describes the dataset and holds the data
    @param batch_keys: which keys to collect from the batch
    @return: a dict containing tensors, an NxD containing the features and rest Nx1 as specified by corresponding
    batch_keys, e.g. targets, batch index etc
    """
    data_module.setup(stage)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def pass_dataloader(dataloader, dataset_attributes_dict=None):
        dl = dataloader

        if dataset_attributes_dict is None:
            dataset_attributes_dict = \
                {
                    'feats': []
                }

            for k in batch_keys[1:]:
                dataset_attributes_dict[k] = []

        # Preallocate the memory, because not doing this results to strange memory leaks
        # Previous implementation utilized python list and attributes from each batch were appended

        first_batch = next(iter(dl))
        images, *rest_info = model.batch_unpack_fn(first_batch, batch_keys)
        feats = model.forward_features(images.to(device))

        n_samples = len(dl.dataset)
        batch_size = feats.shape[0]
        local_attributes = \
            {
                'feats': torch.zeros((n_samples, feats.shape[1]), dtype=feats.dtype)
            }

        for i, k in enumerate(batch_keys[1:]):
            if not isinstance(first_batch[k], torch.Tensor):
                raise NotImplementedError(f'Unsupported type {type(first_batch[k])} of batch element {k}')
            # for now consider all extra attributes of single dimension
            local_attributes[k] = torch.zeros((n_samples,), dtype=first_batch[k].dtype)

        for i, batch in enumerate(tqdm(dl)):
            images, *rest_info = model.batch_unpack_fn(batch, batch_keys)
            feats = model.forward_features(images.to(device))
            local_attributes['feats'][i * batch_size:(i + 1) * batch_size] = feats.detach().cpu()

            # we omit position zero which holds the actual image
            for j, k in enumerate(batch_keys[1:]):
                local_attributes[k][i * batch_size:(i + 1) * batch_size] = rest_info[j]

        for k, v in local_attributes.items():
            dataset_attributes_dict[k].append(local_attributes[k])

        return dataset_attributes_dict

    def concat_dataset_attributes(dataset_attributes_dict):
        for k, v in dataset_attributes_dict.items():
            dataset_attributes_dict[k] = torch.cat(dataset_attributes_dict[k])
        return dataset_attributes_dict

    if stage == 'fit':
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        dataset_attributes = pass_dataloader(train_dataloader, dataset_attributes_dict=None)

        if val_dataloader is not None:
            dataset_attributes = pass_dataloader(val_dataloader, dataset_attributes_dict=dataset_attributes)
        return concat_dataset_attributes(dataset_attributes)
    elif stage == 'test':
        test_dataloader = data_module.test_dataloader()
        if isinstance(test_dataloader, list) and len(test_dataloader) == 2:
            gallery_dataloader, query_dataloader = test_dataloader
            dataset_attributes = \
                {
                    'gallery': concat_dataset_attributes(
                        pass_dataloader(gallery_dataloader, dataset_attributes_dict=None)),
                    'query': concat_dataset_attributes(pass_dataloader(query_dataloader, dataset_attributes_dict=None))
                }
            return dataset_attributes
        elif isinstance(test_dataloader, list):
            raise NotImplementedError('Unsupported type of test dataloader!')
        else:
            dataset_attributes = pass_dataloader(data_module.test_dataloader())
            return concat_dataset_attributes(dataset_attributes)
    else:
        raise AssertionError('Invalid stage.')


@torch.no_grad()
def get_parts_features(model, data_module, parts=('trainval', 'test'), batch_keys=('image', 'target')):
    storage = features_storage.FeaturesStorage(data_module.name)

    if 'trainval' in parts or 'train' in parts:
        # Very ugly
        part_name = 'trainval' if 'trainval' in parts else 'train'

        dataset_attributes = get_features(model, 'fit', data_module, batch_keys=batch_keys)
        storage.add(part_name, dataset_attributes)

    if 'test' in parts:
        dataset_attributes = get_features(model, 'test', data_module, batch_keys=batch_keys)
        if isinstance(dataset_attributes, dict):
            for k, v in dataset_attributes.items():
                storage.add(k, v)
        else:
            storage.add('test', dataset_attributes)

    return storage


def get_features_metric(dataloader, model, batch_keys=None):
    dataset_attributes = \
        {
            'feats': []
        }

    if batch_keys is None:
        # Get the rest available keys by inspecting the batch_unpack function of the model and more specifically the
        # default
        batch_keys = inspect.getfullargspec(model.batch_unpack_fn).defaults[0]

    # Dot not store actual image
    batch_keys = [k for k in batch_keys if k != 'image']

    for k in batch_keys:
        dataset_attributes[k] = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def pass_dataloader(dataloader):
        for batch in dataloader:
            images, *rest_info = model.batch_unpack_fn(batch, keys=['image'] + batch_keys)
            feats = model.forward_features(images.to(device))
            dataset_attributes['feats'].append(feats.detach().cpu())

            for i, k in enumerate(batch_keys):
                dataset_attributes[k].append(rest_info[i])

    with torch.no_grad():
        pass_dataloader(dataloader)

    for k, v in dataset_attributes.items():
        dataset_attributes[k] = torch.cat(dataset_attributes[k])

    return dataset_attributes
