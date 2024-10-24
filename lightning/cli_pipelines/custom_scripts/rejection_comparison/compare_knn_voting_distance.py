import pickle
import torch

input_data = pickle.load(open('input_data.pkl', 'rb'))


def slow_solution(labels, n_labels, indices, k_neighbors, scale=True, distances=None):
    # Use distances as alternative
    n_samples = len(indices)
    nn_distances = torch.zeros((n_samples, n_labels))
    for i in range(n_samples):
        for j in range(k_neighbors):
            # Labels are assumed to be 0...C-1
            label_idx = labels[indices[i][j]]
            # Invert high distance to low weight/score using exp(-x)
            nn_distances[i][label_idx] += torch.exp(-distances[i][j])

        nn_distances[i] -= torch.min(nn_distances[i])
        nn_distances[i] /= torch.max(nn_distances[i])

    weights = nn_distances
    return weights


def fast_solution(labels, n_labels, indices, k_neighbors, scale=True, distances=None):
    # Use distances as alternative
    n_samples = len(indices)
    nn_distances = torch.zeros((n_samples, n_labels))

    label_indices = labels[indices]
    for i in range(n_samples):
        for j in range(k_neighbors):
            # Labels are assumed to be 0...C-1
            label_idx = label_indices[i][j]
            # Invert high distance to low weight/score using exp(-x)
            nn_distances[i][label_idx] += torch.exp(-distances[i][j])


    nn_distances -= torch.stack(nn_distances.shape[1] * [torch.min(nn_distances, 1)[0]]).T
    nn_distances /= torch.stack(nn_distances.shape[1] * [torch.max(nn_distances, 1)[0]]).T


    weights = nn_distances
    return weights


print(torch.all(slow_solution(**input_data) == fast_solution(**input_data)))
