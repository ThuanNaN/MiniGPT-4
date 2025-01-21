eval_mvtec:
	CUDA_VISIBLE_DEVICES=0 \
	python eval_mvtec.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml

train_mvtec:
	CUDA_VISIBLE_DEVICES=0 \
	python train.py --cfg-path train_configs/minigptv2_finetune_mvtec.yaml

demo_mvtec:
	python demo_v2.py --cfg-path eval_configs/minigptv2_eval_mvtec.yaml  --gpu-id 0


eval_textvqa:
	CUDA_VISIBLE_DEVICES=0 \
	python eval_textvqa.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml

train_textvqa:
	CUDA_VISIBLE_DEVICES=0 \
	python train.py --cfg-path train_configs/minigptv2_finetune_textvqa.yaml

demo_textvqa:
	python demo_v2.py --cfg-path eval_configs/minigptv2_eval_textvqa.yaml  --gpu-id 0


train_mix:
	CUDA_VISIBLE_DEVICES=0 \
	python train.py --cfg-path train_configs/minigptv2_finetune_mix.yaml
