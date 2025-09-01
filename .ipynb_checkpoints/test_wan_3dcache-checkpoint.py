import torch
from diffsynth import ModelManager, WanVideo3dCachePipeline, save_video, VideoData
import os
import numpy as np
from PIL import Image
from data.dataset_10k import Dataset10KTestInfo
from cache3d.cache_3d import Cache4D, Cache3D_BufferSelector

model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models(
[
    [
            "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
    ],
    "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    "/root/autodl-tmp/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
])
model_manager.load_lora("/root/autodl-tmp/10_K/lightning_logs/version_3/checkpoints/epoch=9-step=10.ckpt", lora_alpha=1.0)
pipe = WanVideo3dCachePipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

pipe.set_control_adaptor("/root/autodl-tmp/10_K/lightning_logs/version_3/checkpoints/control_adaptor_step_10.ckpt")


# Only `num_frames % 4 == 1` is acceptable
test_path = "/root/autodl-tmp/10_K/10K_1"
test_info = Dataset10KTestInfo(test_path).get_data_dict()

def render_proc(test_info):
    cache_frames = test_info["cache_frames"].unsqueeze(0)
    cache_depth = torch.from_numpy(test_info["cache_depth"]).unsqueeze(0)
    cache_intrinsic = torch.from_numpy(test_info["cache_intrinsic"]).unsqueeze(0)
    cache_w2c = torch.from_numpy(test_info["cache_w2c"]).unsqueeze(0)

    cache = Cache3D_BufferSelector(
        frame_buffer_max=2,
        input_image=cache_frames,
        input_depth=cache_depth,
        input_mask=None,
        input_w2c=cache_w2c,
        input_intrinsics=cache_intrinsic,
        filter_points_threshold=0.05,
        input_format=["B", "N", "C", "H", "W"],
        foreground_masking=False,
    )
    target_w2c = torch.from_numpy(test_info["target_w2c"]).unsqueeze(0)
    target_intrinsic = torch.from_numpy(test_info["target_intrinsic"]).unsqueeze(0)
    render_imgs, render_masks = cache.render_cache(
            target_w2c,
            target_intrinsic,
            start_frame_idx=0,
    )
    test_info["render_imgs"] = render_imgs
    test_info["render_masks"] = render_masks

render_proc(test_info)

video = pipe.infer(
    prompt="清晰的视频",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    all_input=test_info,
    num_inference_steps=50,
    seed=0, tiled=True
)
video_name = "/root/test.mp4"
save_video(video, video_name, fps=15, quality=5)
