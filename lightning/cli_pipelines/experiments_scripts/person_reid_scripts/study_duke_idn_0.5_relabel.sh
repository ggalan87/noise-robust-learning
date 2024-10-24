python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction_w_relabel.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.KNNPairRejection', 'strategy_kwargs': {'k_neighbors': 10, 'rejection_criteria': 'torch_metric_learning.noise_reducers.sample_rejection.HighestIsTheSame', 'rejection_criteria_kwargs': {'threshold': null}, 'use_batched_knn': True, 'with_relabel': True, 'relabel_starting_epoch': 2, 'relabel_confidence': 0.7}, 'keep_only_good_samples': False, 'memory_size': 0, 'warm_up_epochs': 0}" --model.init_args.loss_kwargs="{'margin': 0.2}"

python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction_w_relabel.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.KNNPairRejection', 'strategy_kwargs': {'k_neighbors': 10, 'rejection_criteria': 'torch_metric_learning.noise_reducers.sample_rejection.HighestIsTheSame', 'rejection_criteria_kwargs': {'threshold': null}, 'use_batched_knn': True, 'with_relabel': True, 'relabel_starting_epoch': 10, 'relabel_confidence': 0.6}, 'keep_only_good_samples': False, 'memory_size': 0, 'warm_up_epochs': 0}" --model.init_args.loss_kwargs="{'margin': 0.2}"

python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction_w_relabel.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.KNNPairRejection', 'strategy_kwargs': {'k_neighbors': 10, 'rejection_criteria': 'torch_metric_learning.noise_reducers.sample_rejection.HighestIsTheSame', 'rejection_criteria_kwargs': {'threshold': null}, 'use_batched_knn': True, 'with_relabel': True, 'relabel_starting_epoch': 2, 'relabel_confidence': 0.8}, 'keep_only_good_samples': False, 'memory_size': 0, 'warm_up_epochs': 0}" --model.init_args.loss_kwargs="{'margin': 0.2}"

python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction_w_relabel.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.KNNPairRejection', 'strategy_kwargs': {'k_neighbors': 10, 'rejection_criteria': 'torch_metric_learning.noise_reducers.sample_rejection.HighestIsTheSame', 'rejection_criteria_kwargs': {'threshold': null}, 'use_batched_knn': True, 'with_relabel': True, 'relabel_starting_epoch': 15, 'relabel_confidence': 0.7}, 'keep_only_good_samples': False, 'memory_size': 0, 'warm_up_epochs': 0}" --model.init_args.loss_kwargs="{'margin': 0.2}"

python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction_w_relabel.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.KNNPairRejection', 'strategy_kwargs': {'k_neighbors': 10, 'rejection_criteria': 'torch_metric_learning.noise_reducers.sample_rejection.HighestIsTheSame', 'rejection_criteria_kwargs': {'threshold': null}, 'use_batched_knn': True, 'with_relabel': True, 'relabel_starting_epoch': 2, 'relabel_confidence': 0.6}, 'keep_only_good_samples': False, 'memory_size': 0, 'warm_up_epochs': 0}" --model.init_args.loss_kwargs="{'margin': 0.3}"

python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction_w_relabel.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.KNNPairRejection', 'strategy_kwargs': {'k_neighbors': 10, 'rejection_criteria': 'torch_metric_learning.noise_reducers.sample_rejection.HighestIsTheSame', 'rejection_criteria_kwargs': {'threshold': null}, 'use_batched_knn': True, 'with_relabel': True, 'relabel_starting_epoch': 10, 'relabel_confidence': 0.6}, 'keep_only_good_samples': False, 'memory_size': 0, 'warm_up_epochs': 0}" --model.init_args.loss_kwargs="{'margin': 0.3}"

python extract_feats.py --dataset-name DukeMTMCreID --model-name LitSolider --versions-range 101 106  --epochs-range 38 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

python compute_metrics.py --dataset-name DukeMTMCreID --model-name LitSolider --test-only-accuracy True --self-test False --versions-range 101 106 --epochs-range 38 39 --batched-knn False