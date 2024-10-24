# GBSL - IDN noise levels, v52-54

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5"

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.2"

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.1"

python extract_feats.py --dataset-name Market1501 --model-name LitSolider --versions-list 52 53 54 --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

python compute_metrics.py --dataset-name Market1501 --model-name LitSolider --test-only-accuracy True --self-test False --versions-list 52 53 54 --epochs-range 0 39 --batched-knn False

