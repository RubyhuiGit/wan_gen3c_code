import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideo3dCachePipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import json
import itertools

from data.dataset_10k import Dataset10K
from cache3d.cache_3d import Cache4D, Cache3D_BufferSelector


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideo3dCachePipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
         
        # pixel_values torch.Size([1, 81, 3, 480, 832])
        # target_intrinsic torch.Size([1, 81, 3, 3])
        # target_w2c torch.Size([1, 81, 4, 4])
        # text ['Normal video']
        # data_type ['video']
        # idx torch.Size([1])
        # file_name ['/root/autodl-tmp/10_K/10K_1']
        
        cache_frames = batch["cache_frames"]          #     torch.Size([1, 2, 3, 480, 832])
        cache_depth = batch["cache_depth"]            #     torch.Size([1, 2, 1, 480, 832])
        cache_intrinsic = batch["cache_intrinsic"]    #     torch.Size([1, 2, 3, 3])
        cache_w2c = batch["cache_w2c"]                #     torch.Size([1, 2, 4, 4]) 

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
        target_w2c = batch["target_w2c"]             # torch.Size([1, 81, 3, 3])
        target_intrinsic = batch["target_intrinsic"] # target_w2c torch.Size([1, 81, 4, 4])

        # torch.Size([1, 81, 2, 3, 480, 832]) torch.Size([1, 81, 2, 1, 480, 832])  （-1，1）
        render_imgs, render_masks = cache.render_cache(
            target_w2c,
            target_intrinsic,
            start_frame_idx=0,
            )
        first_frame = batch['pixel_values'][0][0].permute(1, 2, 0).contiguous()
        first_frame = (first_frame * 0.5 + 0.5) * 255    # torch.Size([480, 832, 3])

        # if True:
        #     # 保存原始图像
        #     cur_imgs = batch['pixel_values'][0].clone().detach() # (81, 3, 480, 832)
        #     cur_imgs = (cur_imgs * 0.5 + 0.5) * 255.0
        #     cur_imgs = cur_imgs.permute(0, 2, 3, 1)
        #     view_images = cur_imgs.cpu().numpy().astype(np.uint8)
        #     import imageio
        #     imageio.mimsave(f"{args.output_path}/origin_img.gif", view_images, fps=10)

        #     # 保存render image
        #     cur_render_image = render_imgs[0].clone().detach()    # (81, 2, 3, 480, 832)
        #     cur_render_image = (cur_render_image * 0.5 + 0.5) * 255.0
        #     B, N, C, H, W = cur_render_image.shape
        #     cur_render_image = cur_render_image.reshape(B * N, C, H, W)
        #     cur_render_image = cur_render_image.permute(0, 2, 3, 1)
        #     view_images = cur_render_image.cpu().numpy().astype(np.uint8)
        #     import imageio
        #     imageio.mimsave(f"{args.output_path}/render.gif", view_images, fps=10)

        #     # 保存mask
        #     cur_render_mask = render_masks[0].clone().detach()    # (81, 2, 1, 480, 832)
        #     cur_render_mask = cur_render_mask > 0.5
        #     B, N, C, H, W = cur_render_mask.shape
        #     cur_render_mask = cur_render_mask.reshape(B*N, C, H, W)
        #     cur_render_mask = cur_render_mask.permute(0, 2, 3, 1)
        #     view_mask = (cur_render_mask.cpu().numpy() * 255).astype(np.uint8)     # (162, 480, 832, 1)
        #     view_mask_rgb = np.tile(view_mask, (1, 1, 1, 3))
        #     imageio.mimsave(f"{args.output_path}/render_mask.gif", view_mask_rgb, fps=10)

        #     # 保存clip image
        #     view_clip_img = first_frame.clone().detach()
        #     Image.fromarray(np.uint8(view_clip_img.cpu().numpy())).save(f"{args.output_path}/first_frame.png")

        text, video, path = batch["text"][0], batch["pixel_values"], batch["file_name"][0]
        video = video.permute(0, 2, 1, 3, 4)    # torch.Size([1, 3, 81, 480, 832])

        self.pipe.device = self.device
        if video is not None:
            # 1、prompt 文本编码
            prompt_emb = self.pipe.encode_prompt(text)     # prompt_emb['context']  torch.Size([1, 512, 4096])
            # 2、video vae
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)   # torch.Size([1, 3, 81, 480, 832])
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]          # torch.Size([16, 21, 60, 104])   21是时间维度 16是特征维度

            # 3、首帧编码
            first_frame = Image.fromarray(np.uint8(first_frame.cpu().numpy()))
            _, _, num_frames, height, width = video.shape
            # image_emb["clip_feature"]  torch.Size([1, 257, 1280]) 
            # image_emb["y"] torch.Size([1, 20, 21, 60, 104])
            image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)

            # 4、render image & render mask vae
            mask_pixel_values = []
            masks = []
            for i in range(render_imgs.shape[2]):
                i_render_img = render_imgs[:, :, i, :, : :]           # torch.Size([1, 81, 3, 480, 832])
                i_render_mask = render_masks[:, :, i, :, :, :]        # torch.Size([1, 81, 1, 480, 832])

                print("i", i_render_img.shape, i_render_mask.shape)
                mask_pixel_values.append(i_render_img)
                masks.append(i_render_mask)

            latent_condition = []
            for i, (render_pixel, render_mask) in enumerate(zip(mask_pixel_values, masks)):
                render_mask = render_mask.repeat(1, 1, 3, 1, 1)          # torch.Size([1, 81, 3, 480, 832])

                render_pixel = render_pixel.permute(0, 2, 1, 3, 4)
                render_mask = render_mask.permute(0, 2, 1, 3, 4)
                render_pixel = render_pixel.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)  # torch.Size([1, 3, 81, 480, 832])
                render_mask = render_mask.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)    # torch.Size([1, 3, 81, 480, 832])

                # torch.Size([16, 21, 60, 104])  torch.Size([16, 21, 60, 104])
                mask_pixel_latents = self.pipe.encode_video(render_pixel, **self.tiler_kwargs)[0]
                mask_latents = self.pipe.encode_video(render_mask, **self.tiler_kwargs)[0]

                latent_condition.append(mask_pixel_latents)
                latent_condition.append(mask_latents)
            render_control_latents = torch.cat(latent_condition, dim=0)  # torch.Size([64, 21, 60, 104])
            data = {
                "latents": latents, 
                "prompt_emb": prompt_emb, 
                "image_emb": image_emb, 
                "render_control_latents": render_control_latents}
            
            save_dir = batch["file_name"][0]
            file_path = os.path.join(save_dir, "data.tensors.pth")
            torch.save(data, file_path)



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        json_data = json.load(open(metadata_path))
        self.path = [os.path.join(base_path, frame_info["file_path"], "data.tensors.pth") for frame_info in json_data]
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data
    

    def __len__(self):
        return self.steps_per_epoch


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([dit_path])

        self.pipe = WanVideo3dCachePipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        patch_size = self.pipe.denoising_model().patch_size   # [1, 2, 2]

        self.pipe.init_control_adaptor()

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)     # torch.Size([1, 16, 21, 60, 104]) 
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)  # torch.Size([1, 512, 4096])
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)  # torch.Size([1, 257, 1280])
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)    # torch.Size([1, 20, 21, 60, 104])

        render_latents = batch["render_control_latents"].to(self.device)  # torch.Size([1, 64, 21, 60, 104])
        render_latents_feat = self.pipe.control_adaptor(render_latents)    # torch.Size([1, 5120, 21, 30, 52])

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            control_feat=render_latents_feat,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # 重新写 & debug看保存参数
    def configure_optimizers(self):
        trainable_denoise_params = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        trainable_control_adaptor_params = filter(lambda p: p.requires_grad, self.pipe.get_control_adaptor().parameters())
        all_trainable_params = itertools.chain(trainable_denoise_params, trainable_control_adaptor_params)
        optimizer = torch.optim.AdamW(all_trainable_params, lr=self.learning_rate)
        return optimizer
    
    # 重新写
    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param

        trainable_adaptor_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.get_control_adaptor().named_parameters()))
        trainable_adaptor_param_names = set([named_param[0] for named_param in trainable_adaptor_param_names])
        state_dict = self.pipe.get_control_adaptor().state_dict()
        for name, param in state_dict.items():
            if name in trainable_adaptor_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--dit_files",
        nargs='+',
        help='Paths of DiT files.'
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args

# 数据处理模式
def data_process(args):
    dataset = Dataset10K(
        args.dataset_path,
        video_sample_stride=2,
        video_sample_n_frames=args.num_frames,
        sample_size=(args.height, args.width)
    )
    dataset.__getitem__(0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
# 训练模式
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.json"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_files,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
