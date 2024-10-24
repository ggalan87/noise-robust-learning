# TripletLoss v11 - this runs knn noise reduction on dataset without noise to e.g. infer noise due to occlusions
python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant=null --sampler.init_args.cached_oid_mapping market1501_oid_mapping.pkl

python extract_feats.py --dataset-name Market1501 --model-name LitSolider --versions-list 11 --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

