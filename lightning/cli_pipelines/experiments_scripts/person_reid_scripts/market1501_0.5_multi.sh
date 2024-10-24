# SoftTripleLoss
#python run.py fit --config ./configs/market1501/config_market1501_replicate_pcb.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant=null --sampler.init_args.cached_oid_mapping market1501_clean_oid_mapping.pkl


python run.py fit --config ./configs/market1501/config_market1501_replicate_pcb.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.5" --sampler.init_args.cached_oid_mapping market1501_0.5_oid_mapping.pkl

python run.py fit --config ./configs/market1501/config_market1501_replicate_pcb.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.5" --sampler.init_args.cached_oid_mapping market1501_0.5_oid_mapping.pkl

python run.py fit --config ./configs/market1501/config_market1501_replicate_pcb.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="symmetric_noise_0.5" --sampler.init_args.cached_oid_mapping market1501_0.5_oid_mapping.pkl

