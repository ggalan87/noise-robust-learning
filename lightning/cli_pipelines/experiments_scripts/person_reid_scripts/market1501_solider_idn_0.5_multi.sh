# TripletLoss v18-20 - idn
python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5"

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5"

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="instance_dependent_noise_0.5"

python extract_feats.py --dataset-name Market1501 --model-name LitSolider --versions-list 18 19 20 --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"
