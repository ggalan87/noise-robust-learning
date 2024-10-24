import argparse

import torch
import pickle
from pathlib import Path
from itertools import product
from features_storage import FeaturesStorage
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN, FaissKNN
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from lightning_lite.utilities.seed import seed_everything
from lightning.data.dataset_utils import random_split_perc
from distutils.util import strtobool
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, MulticlassConfusionMatrix, BinaryF1Score

from torchreid.metrics.rank import evaluate_rank
from torchreid.metrics.distance import compute_distance_matrix
from torch_metric_learning.metrics.rank_cylib.rank_cy import eval_from_distmat, eval_from_sorted_indices

import numpy as np

from torch_metric_learning.utils import BatchedKNN


def load_data(cached_path):
    fs = FeaturesStorage(cached_path=cached_path, target_key='target')

    ((raw_training_feats, raw_testing_feats),
     (raw_training_labels, raw_testing_labels),
     (raw_training_cams, raw_testing_cams)) = fs.raw_features()

    return (raw_training_feats, raw_training_labels, raw_training_cams,
            raw_testing_feats, raw_testing_labels, raw_testing_cams)


def run(args):
    dataset_name = args.dataset_name
    target_dataset_name = args.target_dataset_name
    model_class_name = args.model_name
    dm_name = dataset_name.lower()

    if args.versions_range is not None:
        versions = list(range(args.versions_range[0], args.versions_range[1] + 1))
    else:
        versions = args.versions_list

    if args.epochs_range is not None:
        epochs = list(range(args.epochs_range[0], args.epochs_range[1] + 1))
    else:
        epochs = args.epochs_list

    test_only_acc = args.test_only_accuracy
    self_test = args.self_test

    gallery_indices, query_indices = None, None

    seed_everything(13)

    # Override the option above, as self test is done on full test set on its own
    if self_test:
        test_only_acc = True

    for v in versions:
        print(f'Computing accuracies for version {v}...')
        run_path = \
            Path(
                f'./lightning_logs/{dm_name}_{model_class_name}/version_{v}')

        version_accuracies = []

        features_folder_name = 'features' if target_dataset_name is None else f'features-{target_dataset_name.lower()}'
        for e in epochs:
            cached_path = run_path / features_folder_name / f'features_epoch-{e}.pt'

            if not cached_path.exists():
                continue

            #print(cached_path)
            train_embeddings, train_labels, train_cams, test_embeddings, test_labels, test_cams = load_data(cached_path)

            if args.batched_knn:
                distance_fn = LpDistance(normalize_embeddings=False, power=2)
                custom_knn = CustomKNN(distance_fn, batch_size=2048)
                accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1, knn_func=custom_knn, device=torch.device('cpu'))
            else:
                accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

            if test_only_acc:
                if self_test:
                    accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels)

                    acc = accuracies["precision_at_1"]
                elif isinstance(test_embeddings, tuple):
                    gallery_embeddings, query_embeddings = test_embeddings
                    gallery_labels, query_labels = test_labels
                    gallery_cams, query_cams = test_cams

                    query_labels_np = query_labels.detach().cpu().numpy().ravel()
                    gallery_labels_np = gallery_labels.detach().cpu().numpy().ravel()
                    query_cams_np = query_cams.detach().cpu().numpy().ravel()
                    gallery_cams_np = gallery_cams.detach().cpu().numpy().ravel()

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    if dataset_name in ['MSMT17']:

                        if args.batched_knn:
                            distance_fn = LpDistance(normalize_embeddings=False, power=2)
                            print('Running evaluation using BatchedKNN...')
                            knn_func = BatchedKNN(distance_fn, batch_size=2048)
                        else:
                            print('Running evaluation using FAISS...')
                            knn_func = FaissKNN(reset_before=False, reset_after=True)

                        knn_func.train(gallery_embeddings)
                        distances, indices = knn_func(query_embeddings, len(gallery_embeddings))
                        indices_np = indices.numpy()
                        all_cmc, mAP = eval_from_sorted_indices(indices_np,
                                                                query_labels_np,
                                                                gallery_labels_np,
                                                                q_camids=query_cams_np,
                                                                g_camids=gallery_cams_np,
                                                                max_rank=50,
                                                                only_cmc=True)

                    else:
                        # print('Computing distance matrix on feats...', gallery_embeddings.shape, query_embeddings.shape)
                        dists_np = compute_distance_matrix(query_embeddings.to(device), gallery_embeddings.to(device)).detach().cpu().numpy()

                        # print('Computed distance matrix on feats...', gallery_embeddings.shape, query_embeddings.shape)

                        all_cmc, mAP = eval_from_distmat(dists_np,
                                                         query_labels_np,
                                                         gallery_labels_np,
                                                         q_camids=query_cams_np,
                                                         g_camids=gallery_cams_np,
                                                         max_rank=50,
                                                         only_cmc=True)
                        # print(all_cmc, mAP)

                    acc = all_cmc[0]

                else:
                    if gallery_indices is None or query_indices is None:
                        print('Generating indices')
                        gallery_indices, query_indices = random_split_perc(len(test_embeddings), 0.5)

                    accuracies = accuracy_calculator.get_accuracy(
                        test_embeddings[query_indices], test_labels[query_indices], test_embeddings[gallery_indices],
                        test_labels[gallery_indices]
                    )

                    acc = accuracies["precision_at_1"]
            else:
                accuracies = accuracy_calculator.get_accuracy(
                    test_embeddings, test_labels, train_embeddings, train_labels
                )

                acc = accuracies["precision_at_1"]

                # multiclass_confusion_matrix_metric = \
                #     MulticlassConfusionMatrix(num_classes=torch.unique(train_labels), normalize='true')
                #
                # predicted_class_labels =
                # multiclass_confusion_matrix_metric(predicted_class_labels, test_labels)

            version_accuracies.append(acc)
            # print(version_accuracies[-1])

        acc_filename = 'acc.pkl' if target_dataset_name is None else f'acc-{target_dataset_name.lower()}.pkl'
        with open(run_path / acc_filename, 'wb') as f:
            pickle.dump(version_accuracies, f)
        print(version_accuracies)


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-name', required=True, type=str,
                        help='The name of the dataset, or more precisely the class name.')
    parser.add_argument('--target-dataset-name', required=False, type=str,
                        help='The name of the dataset from which to extract features, or more precisely the class name.'
                             'If not specified it will be the same as dataset-name arg.')
    parser.add_argument('--model-name', required=True, type=str,
                        help='The name of the model, or more precisely the model class name.')

    versions_group = parser.add_mutually_exclusive_group(required=True)
    versions_group.add_argument('--versions-list', nargs='+', type=int, help='List of versions')
    versions_group.add_argument('--versions-range', nargs=2, type=int, help='Range of versions (min,max included)')

    epochs_group = parser.add_mutually_exclusive_group(required=True)
    epochs_group.add_argument('--epochs-list', nargs='+', type=int, help='List of epochs')
    epochs_group.add_argument('--epochs-range', nargs=2, type=int, help='Range of epochs (min,max included)')

    parser.add_argument('--test-only-accuracy', required=True, type=strtobool,
                        help='Whether to use only test data for computing accuracy, and split them or use self test')
    parser.add_argument('--self-test', required=True, type=strtobool,
                        help='Whether to use the same data for gallery and query sets.')
    parser.add_argument('--batched-knn', required=True, type=strtobool,
                        help='Whether to use the custom batched knn. Useful for large datasets where FAISS results to'
                             'memory issues.')

    return parser.parse_args()


if __name__ == '__main__':
    run(parse_cli())

