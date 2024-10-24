# Original loss
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --trainer.max_epochs=30 --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"

# MS Loss
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 8054}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"

# SubCenterArcFaceLoss
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 8054}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"

# SoftTripleLoss
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"
python run.py fit --config ./configs/cars/config_cars_replicate.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 8054}" --data.init_args.dataset_args.init_args.training_variant="CARS_0.2noised"

python extract_feats.py --dataset-name Cars --model-name LitInception --versions-range 206 221  --epochs-range 0 30 --batch-keys "image" "target" "target_orig" "data_idx" --parts-list "test"
