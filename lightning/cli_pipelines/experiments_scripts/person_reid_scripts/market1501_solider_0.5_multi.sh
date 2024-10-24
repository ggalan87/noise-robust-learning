# TripletLoss v1-4
#python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant=null --sampler.init_args.cached_oid_mapping market1501_clean_oid_mapping.pkl

#python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.5" --sampler.init_args.cached_oid_mapping market1501_oid_mapping.pkl

#python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.5" --sampler.init_args.cached_oid_mapping market1501_oid_mapping.pkl

python run.py fit --config ./configs/market1501/config_market1501_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.5" --sampler.init_args.cached_oid_mapping market1501_oid_mapping.pkl

