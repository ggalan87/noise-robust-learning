import pickle
from pathlib import Path
import csv
import numpy as np
from visualizations.embeddings_visualization import EmbeddingsVisualizer
from features_storage import FeaturesStorage
from lightning.data.datasets import Market1501DatasetPart
import torch


def compute_pairwise_stats(feats, labels):
    unique_labels = torch.unique(labels)

    means = []
    stds = []

    for l in unique_labels:
        l_feats = feats[labels == l]
        dist_mat = torch.cdist(l_feats, l_feats)
        pairwise_distances = torch.triu(dist_mat, diagonal=1)
        means.append(torch.mean(pairwise_distances))
        stds.append(torch.std(pairwise_distances))

    print(torch.mean(torch.tensor(means)), torch.mean(torch.tensor(stds)))


def load_data(cached_path):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    ((_, raw_testing_feats),
     (_, raw_testing_labels),
     (_, raw_testing_cams)) = fs.raw_features()

    raw_testing_indices = fs.testing_feats[0]['data_idx'], fs.testing_feats[1]['data_idx']

    compute_pairwise_stats(raw_testing_feats[0], raw_testing_labels[0])

    return raw_testing_feats, raw_testing_labels, raw_testing_cams, raw_testing_indices


def construct_label_mask(labels, labels_fraction):
    unique_labels = np.unique(labels)
    n_labels_to_keep = int(labels_fraction * len(unique_labels))
    kept_labels = np.random.choice(unique_labels, n_labels_to_keep, replace=False)

    kept_mask = np.zeros_like(labels, dtype=bool)

    for kl in kept_labels:
        kept_mask = kept_mask | (labels == kl)

    return kept_mask


def construct_occlusion_mask(labels, image_paths):
    # non-occluded
    kept_mask = np.zeros_like(labels, dtype=bool)

    with open('occlusions_in_reid/market1501_missing_lt6.pkl', 'rb') as f:
        occluded_images = pickle.load(f)

    for i, img_path in enumerate(image_paths):
        key = Path(img_path).name
        kept_mask[i] = not occluded_images[key]

    return kept_mask


def construct_fs_for_version(dataset_name, model_class_name, reference_epoch, version):
    dm_name = dataset_name.lower()

    run_path = \
        Path(
            f'/media/amidemo/Data/object_classifier_data/logs/lightning_logs/{dm_name}_{model_class_name}/version_{version}')

    cached_path_reference = run_path / 'features' / f'features_epoch-{reference_epoch}.pt'

    raw_testing_feats, raw_testing_labels, raw_testing_cams, raw_testing_indices =\
        load_data(cached_path_reference)

    gallery_embeddings, query_embeddings = raw_testing_feats
    gallery_labels, query_labels = raw_testing_labels
    gallery_cams, query_cams = raw_testing_cams
    gallery_indices, query_indices = raw_testing_indices

    dataset_attributes = \
        {
            'feats': gallery_embeddings,
            'target': gallery_labels,
            'data_idx': gallery_indices,
        }

    fs = FeaturesStorage(dataset_name)
    fs.add('test', dataset_attributes)

    return fs


dataset_name = 'Market1501'
model_class_name = 'LitSolider'
reference_epoch = 38

fs_orig = construct_fs_for_version(dataset_name, model_class_name, reference_epoch, version=1)
fs_my = construct_fs_for_version(dataset_name, model_class_name, reference_epoch, version=11)

labels = fs_orig.testing_feats['target'].numpy()
kept_mask = construct_label_mask(labels, labels_fraction=0.02)

output_path = '/media/amidemo/Data/object_classifier_data/global_plots'
visualizer = EmbeddingsVisualizer(output_path, dataset_name=dataset_name)
market1501_gallery_dataset = Market1501DatasetPart('/media/amidemo/Data/object_classifier_data/datasets',
                                          part_name='gallery')

images_dict_mask = pickle.load(open('occlusions_in_reid/market1501_gallery_missing_lt6.pkl', 'rb'))

visualizer.add_features_multi({f'{dataset_name}_orig_no': fs_orig, f'{dataset_name}_my_no': fs_my},
                              datasets={'test': market1501_gallery_dataset})
images_server = 'http://amihome2-ubuntu:8347'

kept_mask = visualizer.plot('bokeh', images_server=images_server, kept_mask=kept_mask, images_dict_mask=images_dict_mask)

kept_orig_feats = fs_orig.testing_feats['feats'][kept_mask]
kept_orig_labels = fs_orig.testing_feats['target'][kept_mask]

kept_my_feats = fs_my.testing_feats['feats'][kept_mask]
kept_my_labels = fs_my.testing_feats['target'][kept_mask]

compute_pairwise_stats(kept_orig_feats, kept_orig_labels)
compute_pairwise_stats(kept_my_feats, kept_my_labels)
