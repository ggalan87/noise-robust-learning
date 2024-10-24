#no noise reduction
python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1" --model.init_args.loss_kwargs="{'margin': 0.2}"
#gt noise reduction
python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1" --model.init_args.loss_kwargs="{'margin': 0.2}"
#knn w relabel
python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction_w_relabel.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.KNNPairRejection', 'strategy_kwargs': {'k_neighbors': 10, 'rejection_criteria': 'torch_metric_learning.noise_reducers.sample_rejection.HighestIsTheSame', 'rejection_criteria_kwargs': {'threshold': null}, 'use_batched_knn': True, 'with_relabel': True, 'relabel_starting_epoch': 2, 'relabel_confidence': 0.6}, 'keep_only_good_samples': False, 'memory_size': 0, 'warm_up_epochs': 0}" --model.init_args.loss_kwargs="{'margin': 0.2}"
#knn w/o relabel
python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1" --model.init_args.loss_kwargs="{'margin': 0.2}"
#gbsl w relabel
python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1" --model.init_args.loss_kwargs="{'margin': 0.2}"
#gbsl w/o relabel
python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.GraphBasedSelfLearning', 'strategy_kwargs': {'epochs_frequency': 2, 'with_relabel': False}}" --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1" --model.init_args.loss_kwargs="{'margin': 0.2}"


python extract_feats.py --dataset-name DukeMTMCreID --model-name LitSolider --versions-range 69 74 --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

python compute_metrics.py --dataset-name DukeMTMCreID --model-name LitSolider --test-only-accuracy True --self-test False --versions-range 69 74 --epochs-range 0 39 --batched-knn False
