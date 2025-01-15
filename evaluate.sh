export CUDA_VISIBLE_DEVICES=3
torchrun --master-port 16016 --nproc_per_node 1 eval.py \
 --cfg-path /home/VLAI/thinhpn/STA/MiniGPT-4/eval_configs/minigptv2_benchmark_evaluation.yaml --dataset textvqa