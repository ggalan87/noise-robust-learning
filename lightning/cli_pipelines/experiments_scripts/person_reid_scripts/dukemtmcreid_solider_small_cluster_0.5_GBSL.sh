# TripletLoss v19-20 - GBSL small cluster noise 0.5
python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="small_cluster_noise_0.5"

python run.py fit --config ./configs/dukemtmcreid/config_dukemtmcreid_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.GraphBasedSelfLearning', 'strategy_kwargs': {'epochs_frequency': 2, 'with_relabel': False}}" --data.init_args.dataset_args.init_args.training_variant="small_cluster_noise_0.5"

python extract_feats.py --dataset-name DukeMTMCreID --model-name LitSolider --versions-list 19 20 --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

python compute_metrics.py --dataset-name DukeMTMCreID --model-name LitSolider --test-only-accuracy True --self-test False --versions-range 19 20 --epochs-range 0 39 --batched-knn False
