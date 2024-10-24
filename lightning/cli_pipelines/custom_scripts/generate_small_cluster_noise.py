import torch
from pathlib import Path
import pickle
from lightning_lite.utilities.seed import seed_everything
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import FaissKMeans, FaissKNN
from features_storage import FeaturesStorage

# Small-cluster noise parameters
noise_percentage = 0.25
output_dir = Path('/data/datasets/market1501/noisy_labels')

# noise type
noise_type = 'nn'  # random, nn

# Specify the path of the features
features_root = (
    Path('/media/amidemo/Data/object_classifier_data/logs/lightning_logs/market1501_LitSolider/version_1/features/'))
features_path = features_root / 'features_epoch--1.pt'


def decide_random_label(remaining_labels):
    new_label = remaining_labels[torch.randint(len(remaining_labels), size=(1, ))]
    return new_label


def decide_nn_label(knn_func, kept_training_labels, cluster_embeddings):
    distances, indices = knn_func(cluster_embeddings, 1)

    unique_kept_labels = torch.unique(kept_training_labels)
    label2index = {int(l): i for i, l in enumerate(unique_kept_labels)}
    label_freqs = torch.zeros_like(unique_kept_labels, dtype=torch.int32)
    for i in range(len(indices)):
        suggested_label = kept_training_labels[indices[i]]
        label_freqs[label2index[int(suggested_label)]] += 1

    new_label = unique_kept_labels[torch.argmax(label_freqs)]
    return new_label


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

# Random decision of which labels to remove according to the specified percentage
shuffled_label_indices = torch.randperm(len(unique_labels))
n_labels_to_remove = int(noise_percentage * len(unique_labels))
labels_to_remove = unique_labels[shuffled_label_indices][:n_labels_to_remove]
labels_to_keep = unique_labels[shuffled_label_indices][n_labels_to_remove:]

# Initialize the kmeans algorithm
kmeans_algo = FaissKMeans()

# new_labels
modified_labels = torch.clone(raw_training_labels)

global_keep_mask = torch.zeros_like(raw_training_labels, dtype=torch.bool)

if noise_type == 'nn':
    for l in labels_to_keep:
        global_keep_mask[raw_training_labels == l] = True

    knn_func = FaissKNN(reset_before=False, reset_after=False)
    knn_func.train(raw_training_feats[global_keep_mask])
else:
    knn_func = None


for l in labels_to_remove:
    indices_of_l = torch.where(raw_training_labels == l)[0]
    n_samples = torch.count_nonzero(indices_of_l)

    features_l = raw_training_feats[indices_of_l]
    labels_l = raw_training_labels[indices_of_l]
    cams_l = raw_training_cams[indices_of_l]

    kmeans_result = kmeans_algo(x=features_l, nmb_clusters=int(n_samples / 2))

    cluster_ids = torch.unique(kmeans_result)

    for cluster_id in cluster_ids:
        indices_of_cluster_samples = torch.where(kmeans_result == cluster_id)

        if noise_type == 'random':
            new_label = decide_random_label(labels_to_keep)
        elif noise_type == 'nn':
            global_indices_of_cluster = indices_of_l[indices_of_cluster_samples]
            new_label = decide_nn_label(knn_func, raw_training_labels[global_keep_mask],
                                        raw_training_feats[global_indices_of_cluster])
        else:
            raise NotImplementedError

        modified_labels[indices_of_l[indices_of_cluster_samples]] = new_label

noisy_indices = raw_training_labels != modified_labels

noisy_data = \
    {
        'targets': modified_labels,
        'noisy_indices': noisy_indices
    }

output_path = output_dir / f'small_cluster_noise_{noise_percentage}_{noise_type}.pkl'

with open(output_path, 'wb') as f:
    pickle.dump(noisy_data, f)
