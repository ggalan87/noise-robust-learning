# TripletLoss v1 - no noise
#python run.py fit --config ./configs/msmt17/config_msmt17_replicate_solider.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant=null

# TripletLoss v2-4 - correct class number small cluster noise
#python run.py fit --config ./configs/msmt17/config_msmt17_replicate_solider.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="small_cluster_noise_0.5"

#python run.py fit --config ./configs/msmt17/config_msmt17_replicate_solider.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="small_cluster_noise_0.5"

python run.py fit --config ./configs/msmt17/config_msmt17_replicate_solider.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --data.init_args.dataset_args.init_args.training_variant="small_cluster_noise_0.5"

#python extract_feats.py --dataset-name MSMT17 --model-name LitSolider --versions-list 1 2 3 4  --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"
python extract_feats.py --dataset-name MSMT17 --model-name LitSolider --versions-list 5  --epochs-range 0 39 --batch-keys "image" "target" "data_idx" "camera_id" --parts-list "test"

