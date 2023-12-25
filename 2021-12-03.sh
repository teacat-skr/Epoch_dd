CUDA_VISIBLE_DEVICES=0 python3 dd_kyokawa.py --model resnet18 --label_noise_rate 0.2 --grad_steps 4000
CUDA_VISIBLE_DEVICES=1 python3 dd_kyokawa.py --model resnet18 --pretrained --label_noise_rate 0.2 --grad_steps 4000
