# MVTEC AD dataset
eval_mvtec:
	CUDA_VISIBLE_DEVICES=0 \
	python eval_mvtec.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml

train_mvtec:
	CUDA_VISIBLE_DEVICES=1 \
	python train.py --cfg-path train_configs/minigptv2_finetune_mvtec.yaml

demo_mvtec:
	python demo_v2.py --cfg-path eval_configs/minigptv2_eval_mvtec.yaml --gpu-id 0


# ViVQA dataset
eval_vivqa:
	CUDA_VISIBLE_DEVICES=0 \
	python eval_vivqa.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml

train_vivqa:
	CUDA_VISIBLE_DEVICES=1 \
	python train.py --cfg-path train_configs/minigptv2_finetune_vivqa.yaml

demo_vivqa:
	python demo_v2.py --cfg-path eval_configs/minigptv2_eval_vivqa.yaml --gpu-id 0
