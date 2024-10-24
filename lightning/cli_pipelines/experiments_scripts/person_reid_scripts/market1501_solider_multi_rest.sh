# TripletLoss v61-v66 - symmetric 0.2 and 0.1
python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.2"

#python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.2"

#python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.2"

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1"

#python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1"

#python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_alt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1"


python extract_feats.py --dataset-name Market1501 --model-name LitSolider --versions-range 67 68 --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

python compute_metrics.py --dataset-name Market1501 --model-name LitSolider --test-only-accuracy True --self-test False --versions-range 67 68 --epochs-range 0 39 --batched-knn False
