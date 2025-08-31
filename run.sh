CUDA_VISIBLE_DEVICES="0" python train_wan_3dcache.py \
  --task data_process \
  --dataset_path /root/autodl-tmp/all \
  --output_path /root/autodl-tmp/all \
  --text_encoder_path "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
  --image_encoder_path "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --tiled \
  --num_frames 33 \
  --height 480 \
  --width 832


# CUDA_VISIBLE_DEVICES="0" python train_wan_3dcache.py \
#   --task train \
#   --train_architecture lora \
#   --dataset_path /root/autodl-tmp/all \
#   --output_path /root/autodl-tmp/all \
#   --steps_per_epoch 500 \
#   --max_epochs 10 \
#   --learning_rate 1e-4 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
#   --accumulate_grad_batches 1 \
#   --use_gradient_checkpointing \
#   --dit_files /root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-*.safetensors