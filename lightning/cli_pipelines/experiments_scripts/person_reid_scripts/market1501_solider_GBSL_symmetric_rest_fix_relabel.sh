# TripletLoss v79-v80 - symmetric 0.2 and 0.1, with relabel (fix wrong run)

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.2"

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.1"

python extract_feats.py --dataset-name Market1501 --model-name LitSolider --versions-range 79 80 --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

python compute_metrics.py --dataset-name Market1501 --model-name LitSolider --test-only-accuracy True --self-test False --versions-range 79 80 --epochs-range 0 39 --batched-knn False
