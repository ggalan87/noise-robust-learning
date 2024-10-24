# TripletLoss v81 - GBSL fix
python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gbsl_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="small_cluster_noise_0.25"


python extract_feats.py --dataset-name Market1501 --model-name LitSolider --versions-list 81 --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

python compute_metrics.py --dataset-name Market1501 --model-name LitSolider --test-only-accuracy True --self-test False --versions-range 81 --epochs-range 0 39 --batched-knn False
