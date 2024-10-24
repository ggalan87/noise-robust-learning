import pickle

original = '/data/datasets/market1501/noisy_labels/instance_dependent_noise_0.1.pkl'
updated = '/data/datasets/market1501/noisy_labels/instance_dependent_noise_0.1_updated.pkl'

orig_noise = pickle.load(open(original, 'rb'))
updated_noise = pickle.load(open(updated, 'rb'))
pass