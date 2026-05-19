# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

import os
import math
import copy
import datetime
import torch
import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from typing import Optional, Union, Tuple, List, Dict
from p2p_module import ptp_utils, seq_aligner

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"

# editor base class
class MutualAttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def __call__(self, q, k, v, attention_mask=None, batch_size=None, num_heads=None, scale=None):
        out = self.forward(query=q, key=k, value=v, attention_mask=attention_mask, batch_size=batch_size, heads=num_heads, scale=scale)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1

        return out

    def forward(self, query, key, value, attention_mask=None, batch_size=None, heads=None, scale=None):
        hidden_states = self._memory_efficient_attention_xformers(query=query, key=key, value=value, attention_mask=attention_mask, num_heads=heads)
        # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        hidden_states = hidden_states.to(query.dtype)
        return hidden_states

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def reshape_batch_dim_to_heads(self, tensor, num_heads):
        batch_size, seq_len, dim = tensor.shape
        head_size = num_heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask, num_heads):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states, num_heads)
        return hidden_states


class MutualSelfAttention_p2p(MutualAttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, config, self_replace_steps_p2p=0.2, 
                 start_step=4, end_step=100, start_layer=10, end_layer=16, 
                 sam_masks=None, num_frames=None, 
                 layer_idx=None, step_idx=None, total_steps=50, model_type="SD",
                 # Layout-guided 參數
                 sreg_maps=None,
                 reg_sizes=None,
                 layout_strength=0.3,
                 time_steps=None): 
        super().__init__()
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.sam_masks = sam_masks
        self.num_frames = num_frames
        if type(self_replace_steps_p2p) is float:
            self_replace_steps_p2p = 0, self_replace_steps_p2p
        self.num_self_replace_p2p = (int(config.num_inference_step * self_replace_steps_p2p[0]), 
                                int(config.num_inference_step * self_replace_steps_p2p[1]))
        
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = self.num_self_replace_p2p[1] 
        self.end_step = end_step
        self.start_layer = start_layer
        self.layer_idx = list(range(start_layer, end_layer))
        self.step_idx = list(range(self.start_step, self.end_step)) 
        self.p2p_step_idx = list(range(self.num_self_replace_p2p[0], self.num_self_replace_p2p[1])) 
        
        # Layout-guided 參數
        self.sreg_maps = sreg_maps
        self.reg_sizes = reg_sizes      
        self.layout_strength = layout_strength
        self.time_steps = time_steps

        # self.sreg_maps = None
        # self.reg_sizes = None      
        # self.layout_strength = 0.3
        # self.time_steps = None
        
        print("p2p_MutualSelfAttention at denoising steps: ", self.p2p_step_idx) 
        print("MutualSelfAttention at denoising steps: ", self.step_idx)
        print("MutualSelfAttention at U-Net layers: ", self.layer_idx)

    #TODO: layout-guided modulation function
    def _apply_layout_bias(self, sim: torch.Tensor, frame_idx=0):
        """
        在 softmax 之前應用 layout-guided modulation
        
        Args:
            attn_scores: (1, H*W, H*W) or (num_heads, H*W, H*W)
            H, W: spatial dimensions
            frame_idx: 當前幀的索引
        
        Returns:
            modulated attention scores
        """
        _, spatial_size, _ = sim.shape # sim shape: (num_heads, H*W, H*W)
        if spatial_size not in self.sreg_maps:
            raise ValueError("Spatial size not match to sreg_maps")
        
        mask = self.sreg_maps[spatial_size][frame_idx]     # (1, H*W, H*W)
        size_reg = self.reg_sizes[spatial_size][frame_idx]   # (1, H*W, 1)
        
        # 計算 min/max
        min_value = sim.min(dim=-1, keepdim=True)[0]
        max_value = sim.max(dim=-1, keepdim=True)[0]
        
        # 統一處理為 [batch*heads, spatial, spatial]
        if len(sim.shape) == 4:  # [batch, heads, spatial, spatial]
            print("sim has batch dim")
            sim = rearrange(sim, 'b h s1 s2 -> (b h) s1 s2')

        # 時間衰減係數, λ: 總強度係數
        treg = torch.pow((self.time_steps[self.cur_step] - 1) / 1000, 5)
        lambda_strength = self.layout_strength * treg

        M_pos = (mask > 0) * (1-size_reg) * lambda_strength * (max_value - sim)
        M_neg = ~(mask > 0) * (1-size_reg) * lambda_strength * (sim - min_value)
        
        layout_bias = M_pos - M_neg
        sim = sim + layout_bias.to(sim.dtype)
        return sim
    
    # attn mask：
    def attn_mask_cal(self, masks, num_heads, H, W, dtype):
        masks = masks.masked_fill(masks == 1, float('-inf'))
        masks = masks.expand(1, (H*W), self.num_frames, H, W)
        attention_mask = rearrange(masks, "heads c frame h w -> (frame heads) c (h w)", frame=self.num_frames, h=H, w=W)
        target_attention_mask = torch.zeros_like(attention_mask)
        attention_mask = torch.cat([attention_mask, target_attention_mask], dim=-1)
        
        # 添加 head 維度
        attention_mask = attention_mask.unsqueeze(1)
        
        batch_size, num_heads_dim, seq_q, seq_k = attention_mask.shape
        seq_k_padded = ((seq_k + 7) // 8) * 8
        
        if seq_k_padded != seq_k:
            padded_mask = torch.zeros(
                (batch_size, num_heads_dim, seq_q, seq_k_padded), 
                dtype=dtype, 
                device=attention_mask.device
            )
            padded_mask[:, :, :, :seq_k] = attention_mask
            attention_mask = padded_mask[:, :, :, :seq_k]  # 切片回原始大小但保持alignment

        attention_mask = attention_mask.contiguous().to(dtype)
        return attention_mask
        # masks = masks.masked_fill(masks == 1, float('-inf'))
        # masks = masks.expand(1, (H*W), self.num_frames, H, W)
        # attention_mask = rearrange(masks, "heads c frame h w -> (frame heads) c (h w)", frame=self.num_frames, h=H, w=W)
        # target_attention_mask = torch.zeros_like(attention_mask)
        # attention_mask = torch.cat([attention_mask, target_attention_mask], dim=-1)
        # return attention_mask.unsqueeze(1).to(dtype)
    
    def attn_batch_with_layout(self, q, k, v, masks, num_heads, attention_mask=None):
        H = W = int(math.sqrt(q.shape[1]))
        masks = F.interpolate(masks.float(), size=(H, W), mode='nearest') 
        masks = masks.permute(1, 0, 2, 3).unsqueeze(0) 
        masks = masks.to(k.dtype)

        k_target = k[:, :H*W]
        k_source_bg = k[:, H*W:]
        v_target = v[:, :H*W]
        v_source_bg = v[:, H*W:]

        k_source_bg = rearrange(k_source_bg, "(frame n_head) (h w) c -> n_head c frame h w", frame=self.num_frames, h=H, w=W)
        k_source_bg = k_source_bg * (1 - masks)
        k_source_bg = rearrange(k_source_bg, "n_head c frame h w -> (frame n_head) (h w) c",frame=self.num_frames, h=H, w=W)

        k_target = rearrange(k_target, "(frame n_head) (h w) c -> n_head c frame h w", frame=self.num_frames, h=H, w=W)
        k_target = k_target * masks
        k_target = rearrange(k_target, "n_head c frame h w -> (frame n_head) (h w) c", frame=self.num_frames, h=H, w=W)
        
        key = torch.cat([k_source_bg, k_target], dim=1)
        value = torch.cat([v_source_bg, v_target], dim=1)
        
        # Reshape for multi-head attention
        # attention_mask_converted = self.attn_mask_cal(masks, num_heads, H, W, q.dtype) # attention_mask shape: (B or 1, n_queries, number of keys) 
        batch_size, seq_len, dim = q.shape
        frames = batch_size // num_heads

        # B: batch size(bsz*frames=16), M: sequence length(res:256/1024/4096), H:number of heads(8), K: embeding size per head(160/80/40)
        q = q.reshape(batch_size // num_heads, num_heads, seq_len, dim) # [(B H), M, K]->[B, H, M, K] 
        key = key.reshape(batch_size // num_heads, num_heads, 2*seq_len, dim) # [(B H), M, K]->[B, H, M, K] 
        value = value.reshape(batch_size // num_heads, num_heads, 2*seq_len, dim) # [(B H), M, K]->[B, H, M, K] 

        # === Layout-guided attention with per-frame processing ===
        hidden_states_heads = []
        for head_idx in range(num_heads):
            q_head = q[:, head_idx:head_idx+1, :, :]       
            k_head = key[:, head_idx:head_idx+1, :, :]     
            v_head = value[:, head_idx:head_idx+1, :, :]
            
            scale = 1.0 / math.sqrt(dim)
            attn_scores = torch.matmul(q_head, k_head.transpose(-2, -1)) * scale # (B, 1, M, 2*M)
            
            # Apply layout guidance per frame (only to target part)
            for frame_idx in range(frames):
                attn_scores_target = attn_scores[frame_idx, 0, :, seq_len:] # (seq, seq)
                attn_scores_target = self._apply_layout_bias(attn_scores_target.unsqueeze(0), frame_idx=frame_idx) # (1, M, M)
                attn_scores[frame_idx, 0, :, seq_len:] = attn_scores_target
        
            # Softmax
            attn_probs = F.softmax(attn_scores, dim=-1)
            
            # 計算輸出: (B, 1, Seq, 2*Seq) @ (B, 1, 2*Seq, Dim) -> (B, 1, Seq, Dim)
            hidden_states_head = torch.matmul(attn_probs, v_head) # (batch, 1, M, K)
            hidden_states_heads.append(hidden_states_head)
        
        # Reshape 回原格式
        hidden_states = torch.cat(hidden_states_heads, dim=1)
        hidden_states = hidden_states.reshape(batch_size, seq_len, dim)

        return hidden_states   
    
    def attn_batch(self, q, k, v, masks, num_heads, attention_mask):
        H = W = int(math.sqrt(q.shape[1])) # h*w
        masks = F.interpolate(masks.float(), size=(H, W), mode='nearest') # [f, 1, h, w] 
        masks = masks.permute(1, 0, 2, 3).unsqueeze(0)
        masks = masks.to(k.dtype)

        k_target = k[:, :H*W]
        k_source_bg = k[:, H*W:]
        v_target = v[:, :H*W]
        v_source_bg = v[:, H*W:]

        k_source_bg = rearrange(k_source_bg, "(frame n_head) (h w) c -> n_head c frame h w", frame=self.num_frames, h=H, w=W)
        k_source_bg = k_source_bg * (1-masks)
        k_source_bg = rearrange(k_source_bg, "n_head c frame h w -> (frame n_head) (h w) c", frame=self.num_frames, h=H, w=W)

        # implement to-do, higher priority 
        k_target = rearrange(k_target, "(frame n_head) (h w) c -> n_head c frame h w", frame=self.num_frames, h=H, w=W)
        k_target = k_target * masks
        k_target = rearrange(k_target, "n_head c frame h w -> (frame n_head) (h w) c", frame=self.num_frames, h=H, w=W)
        
        # v_source_bg = rearrange(v_source_bg, "(frame n_head) (h w) c -> n_head c frame h w", frame=self.num_frames, h=H, w=W)
        # v_source_bg = v_source_bg * (1-masks)
        # v_source_bg = rearrange(v_source_bg, "n_head c frame h w -> (frame n_head) (h w) c", frame=self.num_frames, h=H, w=W)
        
        # v_target = rearrange(v_target, "(frame n_head) (h w) c -> n_head c frame h w", frame=self.num_frames, h=H, w=W)
        # v_target = v_target * masks
        # v_target = rearrange(v_target, "n_head c frame h w -> (frame n_head) (h w) c", frame=self.num_frames, h=H, w=W)

        key = torch.cat([k_source_bg, k_target], dim=1) # torch.cat([k_source_fg, k_source_bg, k_target], dim=1)
        value = torch.cat([v_source_bg, v_target], dim=1) # torch.cat([v_source_fg, v_source_bg, v_target], dim=1)
        
        # attention_mask：
        attention_mask_converted = self.attn_mask_cal(masks, num_heads, H, W, q.dtype) # attention_mask shape: (B or 1, n_queries, number of keys) 
        batch_size, seq_len, dim = q.shape
        # B: batch size(bsz*frames=16), M: sequence length(res:256/1024/4096), H:number of heads(8), K: embeding size per head(160/80/40)
        q = q.reshape(batch_size // num_heads, num_heads, seq_len, dim) # [(B H), M, K]->[B, H, M, K]
        q = q.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]
        key = key.reshape(batch_size // num_heads, num_heads, 2*seq_len, dim) # [(B H), M, K]->[B, H, M, K] 
        key = key.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]
        value = value.reshape(batch_size // num_heads, num_heads, 2*seq_len, dim) # [(B H), M, K]->[B, H, M, K] 
        value = value.permute(0, 2, 1, 3) #  [B, H, M, K]->[B, M, H, K]

        # because of mask，calculate hidden states at one time leads to OOM:
        hidden_states_heads = []
        for i in range(num_heads):
            # single head:
            q_single_head = q[:, :, i:i+1, :]  # i-th head
            k_single_head = key[:, :, i:i+1, :]
            v_single_head = value[:, :, i:i+1, :]
            # single head attention
            hidden_states_single_head = xformers.ops.memory_efficient_attention(q_single_head, k_single_head, v_single_head, attn_bias=attention_mask_converted)
            hidden_states_heads.append(hidden_states_single_head)
        
        # reshape
        hidden_states = torch.cat(hidden_states_heads, dim=2)
        hidden_states = hidden_states.permute(0, 2, 1, 3) # [B, M, H, K] -> [B, H, M, K]
        hidden_states = hidden_states.reshape(batch_size, seq_len, dim)  # [B, H, M, K] -> [(B H), M, K]

        return hidden_states
    

    def forward(self, query, key, value, attention_mask, batch_size, heads, scale):
        ## 1. self attn-1 (Sec 3.3)
        if self.cur_step in self.p2p_step_idx and query.shape[1] <= 32 ** 2:
            # softmax, then use controller calculate attention map:
            sim = torch.einsum("b i d, b j d -> b i j", query, key) * scale
            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            uncond_edit, cond_edit, uncond_invert, cond_invert = attn.chunk(4)
            attn_new = torch.cat([uncond_edit, cond_invert.clone(), uncond_invert, cond_invert], dim=0)
            hidden_states = torch.einsum("b i j, b j d -> b i d", attn_new, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states, heads)
            return hidden_states
        #######################################################################################################################################################################
        ## 2. self attn-2 (Sec 3.3)
        elif self.cur_step in self.step_idx and self.cur_att_layer in self.layer_idx:
            query_uncond_new, query_new, query_uncond_invert, query_invert = query.clone().detach().chunk(4)
            key_uncond_new, key_new, key_uncond_invert, key_invert = key.clone().detach().chunk(4)
            value_uncond_new, value_new, value_uncond_invert, value_invert = value.clone().detach().chunk(4) 
            # recon:
            hidden_states_uncond_invert = xformers.ops.memory_efficient_attention(query_uncond_invert, key_uncond_invert, value_uncond_invert, attn_bias=attention_mask)
            hidden_states_invert = xformers.ops.memory_efficient_attention(query_invert, key_invert, value_invert, attn_bias=attention_mask)
            # generate:
            hidden_states_uncond_new = self.attn_batch_with_layout(
                q=query_uncond_new,
                k=torch.cat([key_uncond_new, key_uncond_invert], dim=1),
                v=torch.cat([value_uncond_new, value_uncond_invert], dim=1),
                masks=self.sam_masks,
                num_heads=heads,
                attention_mask=attention_mask
            )
            
            hidden_states_new = self.attn_batch_with_layout(
                q=query_new,
                k=torch.cat([key_new, key_invert], dim=1),
                v=torch.cat([value_new, value_invert], dim=1),
                masks=self.sam_masks,
                num_heads=heads,
                attention_mask=attention_mask
            )
            
            hidden_states = torch.cat([hidden_states_uncond_new, hidden_states_new, hidden_states_uncond_invert, hidden_states_invert], dim=0)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states, heads)
            
            return hidden_states
        ## 3. else
        else:
            return super().forward(query=query, key=key, value=value, 
                                   attention_mask=attention_mask, batch_size=batch_size, heads=heads)


""" modulate Cross-attention with p2p mechanism and layout control """ 
MAX_NUM_WORDS=77

class LocalBlend:
    
    def get_mask(self, x_t, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            print('維度: ', self.alpha_layers.shape[0])
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(x_t, maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts, words, tokenizer, device, config, substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * config.num_inference_step) # NUM_DDIM_STEPS
        self.counter = 0 
        self.th=th

  
class AttentionControl:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        uncond_edit, cond_edit, uncond_invert, cond_invert = attn.clone().detach().chunk(4)
        cond_edit_new = self.forward(torch.cat([cond_edit, cond_invert], dim=0), is_cross, place_in_unet)
        attn = torch.concat([uncond_edit, cond_edit_new[0], uncond_invert, cond_edit_new[1]], dim=0) # cond_invert

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


# Adapted from https://github.com/google/prompt-to-prompt/blob/main/prompt-to-prompt_stable.ipynb
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[2] <= 32 ** 2:
            append_tensor = attn.cpu().detach()
            self.step_store[key].append(copy.deepcopy(append_tensor))
        
        # print('store attention: ', key, len(self.step_store[key]), attn.shape)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            # 每步結束後累加attention map
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]           
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        "divide the attention map value in attention store by denoising steps"       
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        # print('average_attention', {key: len(average_attention[key]) for key in average_attention})
        return average_attention

    def aggregate_attention(self, from_where: List[str], res: int, is_cross: bool, element_name='attn') -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        num_pixels = res ** 2
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                print('is cross',is_cross)
                print('item',item.shape)
                #cross (t,head,res^2,77)
                #self (head,t, res^2,res^2)
                if is_cross:
                    t, h, res_sq, token = item.shape
                    if item.shape[2] == num_pixels:
                        cross_maps = item.reshape(t, -1, res, res, item.shape[-1])
                        out.append(cross_maps)
                else:
                    h, t, res_sq, res_sq = item.shape
                    if item.shape[2] == num_pixels:
                        self_item = item.permute(1, 0, 2, 3) #(t,head,res^2,res^2)
                        self_maps = self_item.reshape(t, h, res, res, self_item.shape[-1])
                        out.append(self_maps)
        out = torch.cat(out, dim=-4)  #average head attention
        out = out.sum(-4) / out.shape[-4]
        return out

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, disk_store=False):
        super(AttentionStore, self).__init__()
        self.disk_store = disk_store
        if self.disk_store:
            time_string = get_time_string()
            path = f'./trash/attention_cache_{time_string}'
            os.makedirs(path, exist_ok=True)
            self.store_dir = path
        else:
            self.store_dir = None
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)

        if ((is_cross and (self.cross_replace_layers[0] <= self.cur_att_layer < self.cross_replace_layers[1])) or 
        (not is_cross and self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1])):
            cond_edit, cond_invert = attn.chunk(2)
            attn_replace = cond_edit.clone().detach().unsqueeze(0)
            attn_base = cond_invert.clone().detach()
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step] # [bsz, 1, 1, max_words], 前期都是1的vector，最後在第45 step才變成0，需要注意
                attn_replace_new = attn_replace
                # attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + attn_replace * (1 - alpha_words)
                if self.prev_controller is not None: 
                # if type(self.prev_controller).__name__ == "AttentionRefine" or self.prev_controller is None:
                    attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + attn_replace * (1 - alpha_words)
                else:
                    attn_replace_new = self.replace_cross_attention(attn_base, attn_replace)
                attn = torch.cat([attn_replace_new.to(attn_base.dtype), attn_base.unsqueeze(0)], dim=0) 
            else:
                # pass
                print("passing here, self-attention replacement")
                attn_replace_new = self.replace_self_attention(attn_base, attn_replace, place_in_unet) 
                attn = torch.cat([attn_replace_new.to(attn_base.dtype), attn_base.unsqueeze(0)], dim=0)
        else:
            cond_edit, cond_invert = attn.chunk(2)
            attn_replace = cond_edit.unsqueeze(0)
            attn_base = cond_invert
            attn = torch.cat([attn_replace, attn_base.unsqueeze(0)], dim=0) 

        return attn
    
    def __init__(self, prompts, tokenizer, device, 
                 num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 cross_replace_layers,
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        self.cross_replace_layers = cross_replace_layers
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

# Adapted from https://github.com/knightyxp/VideoGrain/blob/main/video_diffusion/prompt_attention/attention_util.py
class AttentionControlEditWithLayout(AttentionControlEdit):
    """
    擴展 AttentionControlEdit，支持佈局控制
    
    執行順序：
    1. 存儲 attention (AttentionStore)
    2. 應用佈局控制 (新增)
    3. 應用語義編輯 (replace_cross_attention)
    4. 混合結果 (alpha blending)
    
    核心實現：
    M_cross = R_i ⊙ M_pos - (1-R_i) ⊙ M_neg
    layout_bias = λ * M_cross
    """
    def __init__(self, prompts, tokenizer, device,
                 num_steps: int, 
                 cross_replace_steps: float, 
                 cross_replace_layers,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,
                 controller: Optional[AttentionControlEdit] = None,
                 # Layout 相關參數
                 creg_maps=None, 
                 reg_sizes_c=None, 
                 time_steps=None,
                 layout_end_step=15, 
                 creg=1.0):
        super(AttentionControlEditWithLayout, self).__init__(
            prompts, tokenizer, device, 
            num_steps, cross_replace_steps, 
            cross_replace_layers, self_replace_steps, local_blend
        )
        
        self.creg_maps = creg_maps
        self.reg_sizes_c = reg_sizes_c
        self.time_steps = time_steps
        self.layout_step_idx = list(range(layout_end_step))
        self.creg = creg
        self.current_frame_idx = 0
        self.enable_layout = creg_maps is not None
        self.prev_controller = controller
    
    def replace_cross_attention(self, attn_base, att_replace):
        """
        先應用基礎 controller 的語義編輯，然後應用 Layout 控制
        
        Args:
            attn_base: [heads, spatial, tokens] 基礎 attention（來自 invert）
            att_replace: [batch, heads, spatial, tokens] 待編輯的 attention（來自 edit）
        
        Returns:
            modified_attn: [batch, heads, spatial, tokens]
        """
        if self.prev_controller is not None:
            attn_replace = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        else:
            # 需要確保維度正確：[batch, heads, spatial, tokens]
            if att_replace.dim() == 3:  # [heads, spatial, tokens]
                att_replace = att_replace.unsqueeze(0)
            attn_replace = att_replace
        return attn_replace
    
    def get_layout_bias(self, sim: torch.Tensor, is_cross: bool, place_in_unet: str) -> Optional[torch.Tensor]:
        """
        計算 layout bias (λM_cross)
        
        公式: M_cross = R_i ⊙ M_pos - (1-R_i) ⊙ M_neg
        
        Args:
            sim: [batch*heads, spatial, tokens] attention scores (QK^T / √d)
            is_cross: 是否為 cross-attention
            spatial_size: 空間維度（sequence_length）
        
        Returns:
            layout_bias: [batch*heads, spatial, tokens] 或 None
        """
        if not (is_cross and self.enable_layout and self.cur_step in self.layout_step_idx):
            return None
            # raise ValueError("Layout control is not enabled or conditions not met.")
        
        # batch_size, num_heads, spatial_size, _ = sim.shape # [512(64*8),(4096, 1024, 256, 64),77]
        batch_head, spatial_size, _ = sim.shape
        # print("sim shape:", sim.shape)
        if spatial_size not in self.creg_maps:
            raise ValueError("Spatial size not match to creg_maps")
        
        half_batch = batch_head // 2
        # half_batch = batch_size // 2
        sim_edit = sim[:half_batch]
        sim_invert = sim[half_batch:]
        
        # ================ 3D Tensor sim operation choose first frame =================
        frame_idx = self.current_frame_idx
        num_frames = self.creg_maps[spatial_size].shape[0]
        frame_idx = frame_idx % num_frames if frame_idx >= num_frames else frame_idx

        mask = self.creg_maps[spatial_size][frame_idx]        # [1, spatial, tokens]
        size_reg = self.reg_sizes_c[spatial_size][frame_idx]  # [1, spatial, 1]

        mask = mask.expand(half_batch, -1, -1).to(sim.device, dtype=sim.dtype)                  # [batch*heads, spatial, tokens]
        size_reg = size_reg.expand(half_batch, -1, -1).to(sim.device, dtype=sim.dtype)          # [batch*heads, spatial, 1]
        # ==============================================================================

        # ====================== 4D Tensor sim operation ===============================
        # mask = self.creg_maps[spatial_size]             # [frames, bsz(1), res, 77]
        # size_reg = self.reg_sizes_c[spatial_size]       # [frames, bsz(1), res, 1]
        # mask = mask.repeat(1, 2, 1, 1)                  # [frames, 2, res, 77]
        # size_reg = size_reg.repeat(1, 2, 1, 1)          # [frames, 2, res, 1]
        
        # mask = rearrange(mask, 'f b s t -> (f b) 1 s t')
        # mask = repeat(mask, 'fb 1 s t -> fb h s t', h=num_heads)
        
        # size_reg = rearrange(size_reg, 'f b s d -> (f b) 1 s d')
        # size_reg = repeat(size_reg, 'fb 1 s d -> fb h s d', h=num_heads)
        # ==============================================================================
        
        # # ==========================================
        # # 應用 Layout 約束, 只對 cond_edit 計算 Layout bias
        # # ==========================================
        min_value = sim_edit.min(dim=-1, keepdim=True)[0]
        max_value = sim_edit.max(dim=-1, keepdim=True)[0]
        
        # 時間衰減係數, λ: 總強度係數
        treg = torch.pow((self.time_steps[self.cur_step] - 1) / 1000, 5).item()
        lambda_strength = self.creg * treg
        
        # calculate layout bias
        M_pos = (mask > 0) * (1-size_reg) * lambda_strength * (max_value - sim_edit)
        M_neg = ~(mask > 0) * (1-size_reg) * lambda_strength * (sim_edit - min_value)
        layout_bias = M_pos - M_neg
        
        sim_edit = sim_edit + layout_bias.to(sim.dtype)
        sim = torch.cat([sim_edit, sim_invert], dim=0)
        return sim

def get_equalizer(tokenizer, text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        if len(inds) == 0:
            continue
        equalizer[:, inds] = val
    return equalizer


def make_controller(config, tokenizer, device, 
                    prompts: List[str], is_replace_controller: bool, 
                    cross_replace_steps: Dict[str, float], 
                    cross_replace_layers,
                    self_replace_steps: float, 
                    blend_words=None, equilizer_params=None,
                    # Layout 相關參數
                    creg_maps: Optional[Dict[int, torch.Tensor]] = None,
                    reg_sizes_c: Optional[Dict[int, torch.Tensor]] = None,
                    time_steps: Optional[torch.Tensor] = None,
                    layout_end_step: int = 15,
                    creg: float = 1.0
                    ) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer, device, config)
    
    controller = None        
    if creg_maps is not None:
        print("[INFO] Creating controller with Layout control")
        print(f"[INFO] Base controller: {type(controller).__name__}")
        controller = AttentionControlEditWithLayout(
            prompts=prompts,
            tokenizer=tokenizer,
            device=device,
            num_steps=config.num_inference_step,
            cross_replace_steps=cross_replace_steps,
            cross_replace_layers=cross_replace_layers,
            self_replace_steps=self_replace_steps,
            local_blend=lb,
            controller=controller,
            # Layout 參數
            creg_maps=creg_maps,
            reg_sizes_c=reg_sizes_c,
            time_steps=time_steps,
            layout_end_step=layout_end_step,
            creg=creg
        )
    controller.text_cond_current = None
    return controller   