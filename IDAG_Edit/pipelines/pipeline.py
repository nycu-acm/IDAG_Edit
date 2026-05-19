# Adapted from https://github.com/LPengYang/MotionClone/blob/main/motionclone/pipelines/pipeline_animation.py
import inspect
from typing import Callable, List, Optional, Union, Any, Dict, Tuple
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import omegaconf
from omegaconf import ListConfig
import einops
import imageio
import matplotlib.pyplot as plt
import yaml
import gc  
from ..models.attention import make_controller, MutualSelfAttention_p2p
from ..models.attention_register import regiter_crossattn_editor_diffusers_p2p, regiter_selfattn_editor_diffusers_p2p

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb

from ..utils.xformer_attention import *
from ..utils.conv_layer import *
from ..utils.util import *
from ..utils.util import _in_step, _classify_blocks, ddim_inversion

from .additional_components_mod import *

from torch.optim.adam import Adam
import torch.nn.functional as nnf
from einops import rearrange, repeat
from PIL import Image, ImageDraw
from skimage import measure
from skimage.draw import ellipse
from ..utils.visualization import show_cross_attention_plus_org_img, show_self_attention_comp

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class VideoDirectorPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class VideoDirectorPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[SparseControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.init_latent = ""

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)
        
    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device
            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            # 走這裡
            # print("[INFO]: latents use inverted latents")
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # @torch.no_grad() <== need to calculate gradient, so can not use this decorator
    def recon_guidance_pipe(self,
        video = None,
        config: omegaconf.dictconfig = None,
        save_path = None,
        DDIM_inversion_CFG = True,
        extra_step_kwargs = None
    ):
        # perform DDIM inversion 
        import time
        start_time = time.time()
        # generator = None
        video_latent = self.vae.encode(video.to(self.vae.dtype).to(self.vae.device)).latent_dist.sample(extra_step_kwargs['generator']).clone().detach()
        video_latent = self.vae.config.scaling_factor * video_latent
        video_latent = video_latent.unsqueeze(0)
        video_latent = einops.rearrange(video_latent, "b f c h w -> b c f h w")
        self.init_latent = video_latent
        # print("Recon guidance video shape:", video_latent.shape)  # [1, 4, 14, 64, 64]
        
        print("[INFO]: DDIM inversion...")  
        ddim_latents, context = ddim_inversion_with_context(self, self.scheduler, video_latent, config)

        print("[INFO]: Null-text optimization...") 
        uncond_embeddings, STDG_list, denoised_latent = self.null_optimization(ddim_latents, config, context, save_path, extra_step_kwargs)
        end_time = time.time()
        print("[INFO]: Done Null-text optimization: ", end_time - start_time, "seconds\n")
        # config.null_inner_steps, config.early_stop_epsilon, config.num_inference_step
        
        return ddim_latents, uncond_embeddings, STDG_list, denoised_latent

    # read mask:
    def _read_sam_mask(self, mask_dir, device):
        file_names = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')])
        masks = []

        # mask -> bool
        for file_name in file_names:
            image_path = os.path.join(mask_dir, file_name)
            image = Image.open(image_path).convert('L')  
            image_array = np.array(image)
            # 1->True,0->False
            bool_array = image_array > 0  
            tensor = torch.from_numpy(bool_array).unsqueeze(0).to(device) 
            masks.append(tensor)
        stacked_masks = torch.stack(masks)
        return stacked_masks
        
    # read mask with expanded ellipse:
    def _read_sam_mask_with_ellipse(self, mask_dir, device):
        output_dir = os.path.join(mask_dir, 'expanded_mask')
        os.makedirs(output_dir, exist_ok=True)
        file_names = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        masks = []

        # mask -> bool
        for file_name in file_names:
            image_path = os.path.join(mask_dir, file_name)
            image = Image.open(image_path).convert('L')  
            image_array = np.array(image)
            
            # calculate ellipse:
            _, ellipsoid_mask = self.generate_ellipsoid_mask(image_array, device)
            # merge original mask and ellipsoid_mask:
            merged_mask = np.logical_or(ellipsoid_mask, image_array)
            output_image_path = os.path.join(output_dir, f"merged_{file_name}")
            self.save_ellipse_image(merged_mask, output_image_path)
            merged_tensor = torch.from_numpy(merged_mask).unsqueeze(0).to(device)
            masks.append(merged_tensor)
            
        stacked_masks = torch.stack(masks)  # [N, 1, H, W]
        return stacked_masks
    
    # use scikit-image calculate ellipse mask:
    def generate_ellipsoid_mask(self, image_array, device):
        labeled_mask = measure.label(image_array)
        regions = measure.regionprops(labeled_mask)
        ellipsoid_mask = np.zeros_like(image_array, dtype=np.uint8)

        if len(regions) > 0:
            region = regions[0]

            # calculate ellipse:
            minr, minc, maxr, maxc = region.bbox
            center_y, center_x = region.centroid  # Ellipse center  
            major_axis_length = region.major_axis_length / 2  # Semi-major axis length  
            minor_axis_length = region.minor_axis_length / 2  # Semi-minor axis length  
            orientation = np.degrees(region.orientation)  # Rotation angle, converted to degrees  

            rr, cc = ellipse(int(center_y), int(center_x), 
                             int(major_axis_length), int(minor_axis_length),
                             rotation=np.radians(orientation), shape=image_array.shape)
            ellipsoid_mask[rr, cc] = 1  # fill the ellipse
        ellipsoid_tensor = torch.from_numpy(ellipsoid_mask > 0).unsqueeze(0).to(device)
        return ellipsoid_tensor, ellipsoid_mask  

    def save_ellipse_image(self, ellipsoid_mask, output_image_path):
        image = Image.fromarray((ellipsoid_mask * 255).astype(np.uint8))
        image.save(output_image_path)
    
    # temporal STDG ：
    def calculate_STDG_motion(self, latent_cur_GT, latent_cur, i, step_t, cond_embeddings, config, mask):
        with torch.no_grad():
            # GT for temp_attn (Sec 3.2, Fig 3)
            # cond_embedding，and DDIM inv latents input unet
            _ = get_noise_pred_single(latent_cur_GT, step_t, cond_embeddings, self.unet)
            temp_attn_prob_GT = self.get_temp_attn_prob()

        weight_each_motion = torch.tensor(config.temp_guidance.weight_each).to(latent_cur_GT.dtype).to(latent_cur_GT.device)
        weight_each_motion = torch.repeat_interleave(weight_each_motion/100.0, repeats=6) 
        #  the 100.0 here is only to avoid numberical overflow under float16
        
        latent_cur.requires_grad = True  
        _ = get_noise_pred_single(latent_cur, step_t, cond_embeddings, self.unet) 
        temp_attn_prob = self.get_temp_attn_prob()
        # global temporal guidance:
        loss_motion = compute_temp_loss_with_mask(
            temp_attn_prob_GT, 
            temp_attn_prob, 
            weight_each_motion.detach(), 
            mask
        )
        loss = 100.0*(loss_motion)
        # gradient of loss about latent_cur:
        gradient_motion = torch.autograd.grad(loss, latent_cur, allow_unused=True)[0]
        assert gradient_motion is not None, f"Step {i}: grad is None"
        score_motion = gradient_motion.detach()
        latent_cur.requires_grad = False 
        
        gc.collect() 
        torch.cuda.empty_cache() 
        return score_motion

    # appearance STDG ：
    def calculate_STDG_appearance(self, latent_cur_GT, latent_cur, i, step_t, cond_embeddings, config, sam_mask):
        with torch.no_grad():
            # GT for temp_attn (Sec 3.2, Fig 3)
            # cond_embedding，and DDIM inv latents input unet
            _ = get_noise_pred_single(latent_cur_GT, step_t, cond_embeddings, self.unet) 
            if self.input_config.app_guidance.block_type =="temp": 
                attn_key_GT = self.get_temp_attn_key()
            else:
                attn_key_GT = self.get_spatial_attn1_key()
        
        weight_each_app = torch.tensor(config.app_guidance.weight_each).to(latent_cur_GT.dtype).to(latent_cur_GT.device) 
        if self.input_config.app_guidance.block_type == "temp":
            weight_each_app = torch.repeat_interleave(weight_each_app/100.0, repeats=6) 
        else:
            weight_each_app = torch.repeat_interleave(weight_each_app/100.0, repeats=3)
        latent_cur.requires_grad = True
        _ = get_noise_pred_single(latent_cur, step_t, cond_embeddings, self.unet) 
        if self.input_config.app_guidance.block_type =="temp": 
            attn_key = self.get_temp_attn_key()
        else:
            attn_key = self.get_spatial_attn1_key()
        # global appearance guidance:
        loss_appearance = compute_semantic_loss_with_mask(
            attn_key_GT, 
            attn_key, 
            weight_each_app.detach(),
            sam_mask, 
            block_type=self.input_config.app_guidance.block_type
        )
        loss = 100.0*(loss_appearance) 
        # gradient of loss about latent_cur:
        gradient_appearance = torch.autograd.grad(loss, latent_cur, allow_unused=True)[0] 
        assert gradient_appearance is not None, f"Step {i}: grad is None"

        # STDG = gradient.detach()
        score_appearance = gradient_appearance.detach()

        latent_cur.requires_grad = False 
        
        gc.collect()  
        torch.cuda.empty_cache() 
        return score_appearance

    #TODO: prepare layouts and merged masks
    def _prepare_layouts_and_merged_masks(self, masks: list, dest_size=(64, 64)):
        # Stack layouts: [S, F, 1, H, W]
        layouts = torch.stack(masks, dim=0)
        
        S, N, C, H, W = layouts.shape # N=frame number
        layouts_b = rearrange(layouts, 's f c h w -> (s f) c h w')
        layouts_b_small = F.interpolate(layouts_b.float(), size=dest_size, mode='nearest')
        layouts = rearrange(layouts_b_small, '(s f) c h w -> s f c h w', s=S, f=N) # [S, F, 1, 64, 64]
        
        merged_masks = []
        if S > 1:
            for i in range(N):  # Loop over frames
                merged_mask_frame = torch.sum(layouts[:, i, :, :, :], dim=0)  # [1, 64, 64]
                merged_mask_frame = (merged_mask_frame > 0).to(torch.uint8)
                merged_masks.append(merged_mask_frame)
            merged_masks = torch.stack(merged_masks, dim=0)  # [F, 1, 64, 64]
        else:
            # If only one subject, merged_masks is the same as the single mask
            merged_masks = layouts.squeeze(0)  # [F, 1, 64, 64]
            merged_masks = (merged_masks > 0).to(torch.uint8)
            
        layouts = layouts.permute(1, 0, 2, 3, 4)  # [F, S, 1, 64, 64]
        self.sam_mask['bg'] = ~merged_masks
        return layouts, merged_masks
     
    def null_optimization(self, latents, config, context, save_dir, extra_step_kwargs):
        # assert config is not None, "config is required for FreeControl pipeline"
        if not hasattr(self, 'config'):
            setattr(self, 'input_config', config)
        self.input_config = config
        if not hasattr(self, 'video_name'):
            setattr(self, 'video_name', config.video_path.split('/')[-1].split('.')[0])
        self.video_name = config.video_path.split('/')[-1].split('.')[0]

        self.unet = prep_unet_attention(self.unet)
        self.unet = prep_unet_conv(self.unet)
        #TODO: get mask & layouts: 
        self.sam_mask = {}
        all_masks = []
        for idx, mask_name in enumerate(config.mask_orders):
            item_path = os.path.join(config.mask_dir, mask_name)
            print("[INFO]:",item_path)
            
            if not os.path.exists(item_path):
                raise ValueError("Mask file does not exist: {}".format(item_path))
            
            if config.using_ellipse_mask:
                sam_mask = self._read_sam_mask_with_ellipse(item_path, self.unet.device)
            else:
                sam_mask = self._read_sam_mask(item_path, self.unet.device) # [F, 1, H, W]
            self.sam_mask[mask_name] = sam_mask
            all_masks.append(sam_mask)
        
        self.layouts, self.merged_masks = self._prepare_layouts_and_merged_masks(all_masks, dest_size=(latents[0].shape[3], latents[0].shape[4]))
        # print("[INFO]: merged_masks shape:", self.merged_masks.shape) # [F, 1, 64, 64]
        
        video_length = latents[0].shape[2] # frame
        uncond_embeddings, cond_embeddings = context.chunk(2)
        # multiframe_NT:
        if config.multiframe_NT:
            uncond_embeddings = repeat(uncond_embeddings, 'b n c -> (b f) n c', f=video_length)  
            cond_embeddings = repeat(cond_embeddings, 'b n c -> (b f) n c', f=video_length)  

        uncond_embeddings_list = []
        STDG_list = []

        # STDG = None
        STDG_motion_fore = None
        STDG_motion_back = None
        STDG_appearance_fore = None
        STDG_appearance_back = None
        
        # ========== 新增：初始化 loss 記錄 ==========
        loss_history = []  # 記錄所有 loss
        step_indices = []  # 記錄對應的步數
        current_step = 0
        # ==========================================

        latent_cur = latents[-1]
        bar = tqdm(total=config.null_inner_steps * config.num_inference_step)
        for i in range(config.num_inference_step):
            uncond_embeddings = uncond_embeddings.clone().detach() 
            latent_cur = latent_cur.clone().detach() 
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / float(2*config.num_inference_step)))
            latent_prev = latents[len(latents) - i - 2]
            latent_prev = latent_prev.clone().detach()
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = get_noise_pred_single(latent_cur, t, cond_embeddings, self.unet)
            
            # STDG：
            STDG_dict = {}
            latent_cur_GT = latents[len(latents) - i - 1]
            for mask_name in config.mask_orders:
                sam_mask = self.sam_mask[mask_name]
                STDG_motion_fore = self.calculate_STDG_motion(latent_cur_GT, latent_cur, i, t, cond_embeddings, config, sam_mask)
                STDG_appearance_fore = self.calculate_STDG_appearance(latent_cur_GT, latent_cur, i, t, cond_embeddings, config, sam_mask)
                STDG_fore = (config.STDG_guide[0] * STDG_motion_fore + config.STDG_guide[2] * STDG_appearance_fore)
                STDG_dict[mask_name] = STDG_fore
            STDG_motion_back = self.calculate_STDG_motion(latent_cur_GT, latent_cur, i, t, cond_embeddings, config, self.sam_mask['bg'])
            STDG_appearance_back = self.calculate_STDG_appearance(latent_cur_GT, latent_cur, i, t, cond_embeddings, config, self.sam_mask['bg'])
            STDG_back = (config.STDG_guide[1] * STDG_motion_back + config.STDG_guide[3] * STDG_appearance_back)
            STDG_dict['bg'] = STDG_back
            STDG_list.append(STDG_dict)
                
            for j in range(config.null_inner_steps):
                noise_pred_uncond = get_noise_pred_single(
                    latent_cur, t, 
                    uncond_embeddings, 
                    self.unet
                )
                
                noise_pred = noise_pred_cond + config.cfg_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.scheduler.customized_step_with_grad(
                    noise_pred, t, 
                    latent_cur, 
                    # score=STDG,
                    score_dict=STDG_dict,
                    mask_dict=self.sam_mask,
                    guidance_scale=self.input_config.grad_guidance_scale,
                    indices=[0],
                    **extra_step_kwargs, 
                    return_dict=False
                )[0] 

                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                del latents_prev_rec, noise_pred 
                gc.collect() 
                torch.cuda.empty_cache()  
                optimizer.step()
                assert not torch.isnan(uncond_embeddings.abs().mean())
                loss_item = loss.item()
                # ========== 新增：記錄 loss ==========
                loss_history.append(loss_item)
                step_indices.append(current_step)
                current_step += 1
                # ====================================
                
                bar.update()
                if loss_item < config.early_stop_epsilon + i  * 2e-5:
                    break
            
            for j in range(j + 1, config.null_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings.detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])

                latents_input = torch.cat([latent_cur] * 2)
                assert context is not None
                noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_prediction_text + config.cfg_scale * (noise_prediction_text - noise_pred_uncond)
                latent_cur = self.scheduler.customized_step(
                    noise_pred, t, 
                    latent_cur, 
                    # score=STDG,
                    score_dict=STDG_dict,
                    mask_dict=self.sam_mask,
                    guidance_scale=self.input_config.grad_guidance_scale,
                    indices=[0],
                    **extra_step_kwargs, 
                    return_dict=False
                )[0].detach()
        bar.close()
        return uncond_embeddings_list, STDG_list, latent_cur    

    # Adapted from https://github.com/knightyxp/VideoGrain/blob/main/video_diffusion/pipelines/ddim_spatial_temporal.py
    def _prepare_attention_layout(
        self,
        bsz,
        height,
        width,
        layouts: torch.Tensor,          
        prompts: List[str],
        device: torch.device,
        use_cross_frame_layout: bool = False,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        準備 self-attention 和 cross-attention 所需的佈局映射
        1. text_cond: [2*bsz, 77, 768] (uncond + cond embeddings)
        2. creg_maps: {res: [frames, bsz, res, 77]} 空間-文字對齊遮罩
        3. reg_sizes_c: {res: [frames, bsz, res, 1]} 未佔用空間比例
        4. sreg_maps: {res: [frames, 1, res, res]} Self-attention 佈局遮罩
        5. reg_sizes: {res: [frames, 1, res, 1]} Self-attention 正則化大小
        
        Args:
            bsz: batch size
            height, width: latent size
            layouts: [frames, num_regions, 1, H, W]
            prompts: [global prompt, local1, local2, ...]
        
        Returns:
            text_cond: 文本條件 embedding
            creg_maps: cross-attention 遮罩字典
            reg_sizes_c: cross-attention 區域大小字典
            sreg_maps: self-attention 遮罩字典
            reg_sizes: self-attention 區域大小字典
        """
       
        # ===============================================
        # step 1: generate global/local text embeddings 
        # ===============================================
        text_input = self.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                    max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        cond_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        
        # ===============================================
        # Step 2: Build Per-Word-Per-Pixel Maps
        # ===============================================
        frames = layouts.shape[0]
        num_regions = layouts.shape[1]
        pww_maps = torch.zeros(frames, 1, 77, height, width, device=device)
        global_tokens = text_input.input_ids[0]
        
        for region_idx, local_prompt in enumerate(prompts[1:], start=1):
            local_length = text_input.length[region_idx] - 2  # 去掉 BOS/EOS
            # print(f"[DEBUG] Processing region {region_idx}: '{local_prompt}' with length {local_length}")
            local_tokens = text_input.input_ids[region_idx, 1:1+local_length]
            
            # 在全局提示中找到匹配位置
            for global_pos in range(77 - local_length + 1):
                global_segment = global_tokens[global_pos:global_pos+local_length]
                
                if torch.equal(global_segment, local_tokens):
                    for frame_idx in range(frames):
                        pww_maps[frame_idx, :, global_pos:global_pos+local_length, :, :] = \
                            layouts[frame_idx, region_idx-1:region_idx, :, :, :]
                    
                    # global prompt embedding 替換成 local prompt embedding
                    cond_embeddings[0, global_pos:global_pos+local_length] = cond_embeddings[region_idx, 1:1+local_length]
                    print(f"[INFO] Region {region_idx} ('{local_prompt}') → token pos {global_pos}:{global_pos+local_length}")
                    break
        
        # ==============================================================
        # step 3: bulid creg_maps/sreg_maps & reg_sizes/reg_sizes_c
        # ==============================================================
        creg_maps = {}
        sreg_maps = {}
        reg_sizes = {}
        
        layouts_reshaped = rearrange(self.layouts, 'f s c h w -> (f s) c h w')
        for r in range(4):
            h = height // (2 ** r)
            w = width // (2 ** r)
            res = h * w

            # 調整佈局到當前解析度: [F, S, 1, 64, 64] -> [F, S, 1, h, w]
            layouts_b_small = F.interpolate(layouts_reshaped.float(), size=(h,w), mode='nearest')
            layouts = rearrange(layouts_b_small, '(f s) c h w -> f s c h w', s=self.layouts.shape[1], f=self.layouts.shape[0])
            
            # ---------------------------------------
            # Cross-Attention: 逐 frame 處理
            # ---------------------------------------
            creg_all_frames = []
            pww_maps_reshaped = rearrange(pww_maps, 'f c t h w -> (f c) t h w')
            pww_maps_small = F.interpolate(pww_maps_reshaped, size=(h, w), mode='nearest')
            pww_maps_at_res = rearrange(pww_maps_small, '(f c) t h w -> f c t h w', f=frames)
            
            for frame_idx in range(frames):
                pww_frame = pww_maps_at_res[frame_idx, 0]  # [77, h, w]
                creg_frame = pww_frame.view(77, -1).t()  # [res, 77]
                creg_frame = creg_frame.unsqueeze(0).expand(bsz, -1, -1) # [bsz, res, 77]
                creg_all_frames.append(creg_frame)

            creg_maps[res] = torch.stack(creg_all_frames, dim=0)  # [frames, bsz, res, 77]
                
            # ---------------------------------------
            # Self-Attention: 逐 frame 處理 (region-aware)
            # ---------------------------------------
            if use_cross_frame_layout:
                # flatten all of frames: [num_regions, 1, frames*res]
                layouts_flat = rearrange(layouts, 'f s c h w -> s c (f h w)')
                
                # Spatio-Temporal Outer Product
                layout_vec = layouts_flat.view(num_regions, -1, 1)      # [S, total_tokens, 1]
                layout_vec_t = layouts_flat.view(num_regions, 1, -1)    # [S, 1, total_tokens]
                region_masks = layout_vec * layout_vec_t                # [S, total_tokens, total_tokens]
                
                # 加總所有區域
                sreg_combined = region_masks.sum(0).unsqueeze(0).repeat(bsz, 1, 1)  # [bsz, total_tokens, total_tokens]
                sreg_maps[res] = sreg_combined
                
                # 計算每個 token 的空白比例
                occupied_mask = layouts_flat.sum(dim=0).clamp(0, 1)  # [1, total_tokens]
                reg_size_per_token = 1.0 - occupied_mask.squeeze(0)  # [total_tokens]
                reg_sizes[res] = reg_size_per_token.unsqueeze(0).unsqueeze(-1).repeat(bsz, 1, 1) # [bsz, total_tokens, 1]
            else:
                sreg_all_frames = []
                reg_sizes_all_frames = []
                
                for frame_idx in range(frames):
                    layout_frame = layouts[frame_idx]  # [num_regions, 1, h, w]

                    # 為每個 pixel 分配所屬區域的 ID（0=背景, 1=區域1, 2=區域2...）
                    region_id_map = torch.zeros(h, w, device=device)
                    for region_idx in range(num_regions):
                        region_mask = layout_frame[region_idx, 0] # [h, w]
                        region_id_map = torch.where(
                            region_mask > 0.5,
                            torch.full_like(region_id_map, region_idx + 1),
                            region_id_map
                        )
                    
                    region_id_flat = region_id_map.view(-1)  # [res]

                    # sreg[i, j] = 1 if region_id[i] == region_id[j] and != 0
                    same_region = (region_id_flat.unsqueeze(1) == region_id_flat.unsqueeze(0)) # [res, res]
                    non_background = (region_id_flat != 0) # [res]

                    sreg_frame = same_region.float() * non_background.unsqueeze(1).float() * non_background.unsqueeze(0).float()
                    sreg_frame = sreg_frame.unsqueeze(0).repeat(bsz, 1, 1)  # [bsz, res, res]
                    sreg_all_frames.append(sreg_frame)
                                        
                    # Regularization size
                    background = (region_id_flat == 0).float()  # [res]
                    reg_size_frame = background.unsqueeze(-1)  # [res, 1]
                    reg_size_frame = reg_size_frame.unsqueeze(0).repeat(bsz, 1, 1)  # [bsz, res, 1]
                    reg_sizes_all_frames.append(reg_size_frame)
                
                sreg_maps[res] = torch.stack(sreg_all_frames, dim=0)  # [frames, bsz, res, res]
                reg_sizes[res] = torch.stack(reg_sizes_all_frames, dim=0)  # [frames, bsz, res, 1]
                
        # ==========================================
        # step 4: prepare for text_cond_emb
        # ==========================================
        text_cond = cond_embeddings[:1].repeat(bsz,1,1)
        return text_cond, creg_maps, sreg_maps, reg_sizes
    
    # excute multiple object editing
    def editing_pipe(
        self,
        video,
        ddim_latents, 
        uncond_embeddings, 
        generator, 
        config, 
        STDG_list,
        save_dir,
        eta: float = 0.0,
    ):
        with torch.no_grad():
            assert uncond_embeddings is not None, "editing_pipe() needs uncond_embeddings!!!"
            batch_size = 1
            num_videos_per_prompt = 1
            num_channels_latents = self.unet.in_channels
            device = self._execution_device
            negative_prompt = config.negative_prompt
            vis_cross_attn = config.vis_cross_attn

            # perform classifier_free_guidance in default
            cfg_scale = config.cfg_scale or 7.5
            do_classifier_free_guidance = True

            # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
            with_uncond_embedding = do_classifier_free_guidance if uncond_embeddings is None else False
            video_length = ddim_latents[-1].shape[2]
            # original text embedding:  torch.Size([32, 77, 768])
            invert_text_embeddings = self._encode_prompt(
                config.inversion_prompt, 
                device, 
                num_videos_per_prompt, 
                True, 
                negative_prompt,
            )
            
            # target text embedding
            new_prompt = OmegaConf.to_container(config.new_prompt[0], resolve=True)
            edit_prompt = str(new_prompt[0]) if isinstance(new_prompt, list) else str(new_prompt)
            current_text_embeddings = self._encode_prompt(
                edit_prompt, 
                device, 
                num_videos_per_prompt, 
                with_uncond_embedding, 
                negative_prompt,
            )
            
            # for blending match to latent
            latent_mask = self.merged_masks
            latent_mask = latent_mask.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 1, F, H, W]
            latent_mask = latent_mask.to(invert_text_embeddings.dtype)

            # Initialize latent for current object
            current_latent = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                config.H,
                config.W,
                current_text_embeddings.dtype,
                device,
                generator,
                ddim_latents[-1],
            )
            
            # Prepare timesteps
            self.scheduler.set_timesteps(config.num_inference_step, device=device)
            timesteps = self.scheduler.timesteps
            
            #============  start of prepare attention layout for current object & setup attention mechanism ===============#
            _, _, _, downsample_height, downsample_width = ddim_latents[-1].shape
            text_cond, creg_maps, sreg_maps, reg_sizes = self._prepare_attention_layout(batch_size, downsample_height,downsample_width, 
                                                                                        self.layouts, new_prompt, device)
            # setting text embedding for multiframe_NT
            if config.multiframe_NT:
                current_invert_embeddings = repeat(invert_text_embeddings, 'b n c -> (b f) n c', f=video_length)  
                current_text_embeddings = repeat(current_text_embeddings, 'b n c -> (b f) n c', f=video_length)
                layout_text_cond = repeat(text_cond, 'b n c -> (b f) n c', f=video_length)
            else:
                current_invert_embeddings = invert_text_embeddings
            
            # Setup P2P controller for current object
            prompts = [config.inversion_prompt, edit_prompt]
            cross_replace_steps = {'default_': config.p2p_cross_replace_steps,}
            cross_replace_layers = config.p2p_cross_replace_layers
            self_replace_steps = config.p2p_self_replace_steps
            
            if config.p2p_blend_word_base is not None and config.p2p_blend_word_new is not None:
                blend_word = (((config.p2p_blend_word_base,), (config.p2p_blend_word_new,)))
            else:
                blend_word = None
            eq_params = {"words": tuple(config.p2p_eq_params_words), "values": tuple(config.p2p_eq_params_values)}
            controller = make_controller(config, self.tokenizer, self.unet.device, 
                                        prompts, config.p2p_cross_is_replace_controller, 
                                        cross_replace_steps, 
                                        cross_replace_layers, 
                                        self_replace_steps, blend_word, eq_params,
                                        # Layout 參數
                                        creg_maps=creg_maps,
                                        reg_sizes_c=reg_sizes,
                                        time_steps=timesteps,
                                        layout_end_step=40,
                                        creg=1.0)
            regiter_crossattn_editor_diffusers_p2p(self.unet, controller)
            
            # Setup self-attention editor
            SELF_START_STEP = self.input_config.MutualSelfAttn_steps[0]
            SELF_END_STEP = self.input_config.MutualSelfAttn_steps[1]
            SELF_START_LAYER = self.input_config.MutualSelfAttn_layers[0]
            SELF_END_LAYER = self.input_config.MutualSelfAttn_layers[1]
            selfattn_editor_p2p = MutualSelfAttention_p2p(config, self_replace_steps_p2p=self_replace_steps,
                                                        start_step=SELF_START_STEP,end_step=SELF_END_STEP, 
                                                        start_layer=SELF_START_LAYER, end_layer=SELF_END_LAYER,
                                                        sam_masks=self.merged_masks, num_frames=video_length,
                                                        # Layout 參數
                                                        sreg_maps=sreg_maps,
                                                        reg_sizes=reg_sizes,
                                                        layout_strength=0.3,
                                                        time_steps=timesteps)
            regiter_selfattn_editor_diffusers_p2p(self.unet, selfattn_editor_p2p)
            #============ End of prepare attention layout for current object & setup attention mechanism ===================#
            
            # Denoising loop for current object
            with self.progress_bar(total=config.num_inference_step) as progress_bar:
                if uncond_embeddings is not None:
                    start_time = config.num_inference_step
                    assert (timesteps[-start_time:] == timesteps).all()
                
                for i, step_t in enumerate(timesteps):
                    invert_latent = ddim_latents[len(ddim_latents)-i-1]
                    latent_model_input = torch.cat([torch.cat([current_latent] * 2), torch.cat([invert_latent] * 2)]) if do_classifier_free_guidance else current_latent
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_t)
                    text_embeddings_input = torch.cat([torch.cat([uncond_embeddings[i], current_text_embeddings]), current_invert_embeddings])
                    controller.text_cond_current = torch.cat([torch.cat([uncond_embeddings[i], layout_text_cond]), current_invert_embeddings])
                    
                    noise_pred = self.unet(
                        latent_model_input, 
                        step_t, 
                        encoder_hidden_states=text_embeddings_input,
                    ).sample.to(dtype=current_latent.dtype)
                    
                    new_noise_pred = noise_pred[[1]] + cfg_scale * (noise_pred[[1]] - noise_pred[[0]]) # cfg 
                    
                    if len(STDG_list) > 0:
                        current_latent = self.scheduler.customized_step(new_noise_pred, step_t, current_latent,
                                                        # score=STDG_list[i],
                                                        score_dict=STDG_list[i],
                                                        mask_dict=self.sam_mask,
                                                        guidance_scale=self.input_config.grad_guidance_scale,
                                                        indices=[0],
                                                        **extra_step_kwargs, return_dict=False)[0].detach()
                    else:
                        # pass
                        current_latent = self.scheduler.customized_step(new_noise_pred, step_t, current_latent, score=None,
                                                        guidance_scale=self.input_config.grad_guidance_scale,
                                                        indices=[0], 
                                                        **extra_step_kwargs, return_dict=False)[0].detach()
                    
                    # noise_source_latents = self.scheduler.add_noise(
                    #     self.init_latent, torch.randn_like(invert_latent), step_t
                    # )
                    # current_latent = current_latent * latent_mask + noise_source_latents * (1 - latent_mask)
                    
                    if vis_cross_attn:
                        save_path = os.path.join(save_dir, os.path.join('visualization_denoise', "mutliple_objects"))
                        os.makedirs(save_path, exist_ok=True)
                        attention_output = show_cross_attention_plus_org_img(step_t, self.tokenizer, edit_prompt, video, controller, 32, ["up","down"], save_path=save_path)
                    
                    progress_bar.update()
                
                # Clean up attention editors for next object
                self._clean_attention_editors()
            
            return current_latent

    def _clean_attention_editors(self):
        """Clean up attention editors between object processing"""
        # Reset attention hooks and controllers
        for name, module in self.unet.named_modules():
            if hasattr(module, 'attention_editor'):
                delattr(module, 'attention_editor')
            if hasattr(module, 'controller'):
                delattr(module, 'controller')

    # https://github.com/LPengYang/MotionClone/blob/main/motionclone/utils/motionclone_functions.py
    def get_temp_attn_prob(self,index_select=None):

        attn_prob_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "VersatileAttention" in module_name and _classify_blocks(self.input_config.temp_guidance.blocks, name):
                key = module.processor.key
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                key = module.reshape_heads_to_batch_dim(key).contiguous()
                
                query = module.processor.query
                if index_select is not None:
                    query = query[index_picked]
                query = module.reshape_heads_to_batch_dim(query).contiguous()
                

                attention_probs = module.get_attention_scores(query, key, None)         
                attention_probs = attention_probs.reshape(-1, module.heads,attention_probs.shape[1], attention_probs.shape[2])
                
                attn_prob_dic[name] = attention_probs

        return attn_prob_dic
    
    def get_temp_attn_key(self,index_select=None):

        attn_key_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "VersatileAttention" in module_name and _classify_blocks(self.input_config.app_guidance.blocks, name):
                key = module.processor.key
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                
                attn_key_dic[name] = key

        return attn_key_dic
        
    def get_spatial_attn1_key(self, index_select=None):
        attn_key_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "Attention" in module_name and 'attn1' in name and 'attentions' in name and _classify_blocks(self.input_config.app_guidance.blocks, name):
                key = module.processor.key
                # [64,256,1280]
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                
                attn_key_dic[name] = key
                # [frame, H*W, head*dim] [16,256,1280]

        return attn_key_dic

