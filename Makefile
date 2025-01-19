eval_mvtec:
	CUDA_VISIBLE_DEVICES=0 \
	python eval_mvtec.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml

train_mvtec:
	CUDA_VISIBLE_DEVICES=0 \
	python train.py --cfg-path train_configs/minigptv2_finetune_mvtec.yaml


eval_textvqa:
	CUDA_VISIBLE_DEVICES=0 \
	python eval_textvqa.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml

train_textvqa:
	CUDA_VISIBLE_DEVICES=0 \
	python train.py --cfg-path train_configs/minigptv2_finetune_textvqa.yaml
