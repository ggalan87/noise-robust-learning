import torch
from pathlib import Path
import pickle
from lightning_lite.utilities.seed import seed_everything
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import FaissKMeans, FaissKNN
from features_storage import FeaturesStorage

# Small-cluster noise parameters
dataset_name = 'msmt17'
noise_percentage = 0.5
output_dir = Path(f'/data/datasets/{dataset_name}/noisy_labels')

# noise type
noise_type = 'nn'  # random, nn

# Specify the path of the features
features_root = (
    Path(f'/media/amidemo/Data/object_classifier_data/logs/lightning_logs/{dataset_name}_LitSolider/version_0/features/'))
features_path = features_root / 'features_epoch--1.pt'


# seed everything
seed_everything(13)

# Load the features
fs = FeaturesStorage(cached_path=features_path)
((raw_training_feats, _), (raw_training_labels, _), (raw_training_cams, _)) = fs.raw_features()

# IMPORTANT! Get the sorted versions of the above such that they are not affected by possible shuffle in dataloader
# The order is the one provided by list(Path(images_folder).rglob('*.jpg'))
# IMPORTANT - 2! The labels are extracted after any possible relabeling to 0 - numclasses-1
sorted_indices = torch.argsort(fs.training_feats['data_idx'])
raw_training_feats = raw_training_feats[sorted_indices]
raw_training_labels = raw_training_labels[sorted_indices]
raw_training_cams = raw_training_cams[sorted_indices]

# Get the unique labels
unique_labels = torch.unique(raw_training_labels)

# Random decision of which dataset indices to change / keep. This is done per label and not globally
indices_to_change = []
indices_to_keep = []
for l in unique_labels:
    indices_of_l = torch.where(raw_training_labels == l)[0]
    shuffled_l_indices = torch.randperm(len(indices_of_l))
    n_indices_of_l_to_change = max(1, int(noise_percentage * len(indices_of_l)))
    indices_to_change.append(indices_of_l[shuffled_l_indices[:n_indices_of_l_to_change]])
    indices_to_keep.append(indices_of_l[shuffled_l_indices[n_indices_of_l_to_change:]])

indices_to_change = torch.cat(indices_to_change)
indices_to_keep = torch.cat(indices_to_keep)

# new_labels
modified_labels = torch.clone(raw_training_labels)

knn_func = FaissKNN(reset_before=False, reset_after=False)
knn_func.train(raw_training_feats[indices_to_keep])

# Decide per label such that all rest labels are considered for label flip
for l in unique_labels:
    indices_of_l_to_change = torch.where(raw_training_labels[indices_to_change] == l)[0]

    all_indices_of_l = torch.where(raw_training_labels == l)[0]

    if len(indices_of_l_to_change) == len(all_indices_of_l):
        print(f'Omit relabel of any sample of class {l} in order to avoid to remove all its '
              f'{len(indices_of_l_to_change)} samples!')
        continue

    n_samples = torch.count_nonzero(indices_of_l_to_change)

    # n_changed += len(indices_of_l_to_change)

    # Find at least len+1 neighbors such that at least one is from another label
    n_indices_of_l_to_keep = len(all_indices_of_l) - len(indices_of_l_to_change)
    k_neighbors = n_indices_of_l_to_keep + 1
    distances, indices = knn_func(raw_training_feats[indices_of_l_to_change], k_neighbors)

    for i, sample_neighbors in enumerate(indices):
        for idx in sample_neighbors:
            nn_label = raw_training_labels[indices_to_keep][idx]
            if nn_label != l:
                new_label = nn_label
                break
        else:
            print(len(indices_of_l_to_change), len(all_indices_of_l), k_neighbors)
            raise AssertionError('Logic error, should not have reached here')

        modified_labels[indices_to_change[indices_of_l_to_change[i]]] = new_label


noisy_indices = raw_training_labels != modified_labels

# if len(torch.unique(modified_labels)) != len(unique_labels):
#     raise AssertionError('Need to reconsider implementation for this dataset because at least a class is missing!')

noisy_data = \
    {
        'targets': modified_labels,
        'noisy_indices': noisy_indices
    }

print(f'Changed {noise_percentage} - {torch.count_nonzero(noisy_indices).item()}')

output_path = output_dir / f'instance_dependent_noise_{noise_percentage}.pkl'

with open(output_path, 'wb') as f:
    pickle.dump(noisy_data, f)
