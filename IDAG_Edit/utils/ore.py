import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np


def get_single_prompt_embeds(prompt, tokenizer, text_encoder, device):
    """Get embeddings for a single prompt"""
    embeds_max_length = tokenizer.model_max_length
    text_inputs_id = tokenizer(
        prompt,
        padding="max_length",
        max_length=embeds_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    
    prompt_embeds = text_encoder(text_inputs_id.to(device))[0]
    return prompt_embeds


def find_sublist_indices(A, B):
    """Find indices where sublist B appears in list A"""
    len_a, len_b = len(A), len(B)
    indices = []
    
    for i in range(len_a - len_b + 1):
        if A[i : i + len_b] == B:
            indices = list(range(i, i + len_b))
            break
    
    return indices


def get_substituted_base_embeds(base_prompt, target_noun_list, tokenizer, text_encoder, device):
    """
    Substitute target nouns in base prompt with their individual embeddings
    """
    base_prompt_embeds = get_single_prompt_embeds(base_prompt, tokenizer, text_encoder, device)
    base_ids = tokenizer(base_prompt).input_ids
    
    for noun_chunk in target_noun_list:
        target_ids = tokenizer(noun_chunk).input_ids[1:-1]  # Remove BOS/EOS
        indices = find_sublist_indices(base_ids, target_ids)
        
        if indices:
            # Get embedding for the noun chunk alone
            noun_embeds = get_single_prompt_embeds(noun_chunk, tokenizer, text_encoder, device)
            # Substitute only the matching positions
            base_prompt_embeds[0, indices, :] = noun_embeds[0, indices, :]
    
    return base_prompt_embeds


def get_object_restricted_embeds(base_prompt, target_noun_list, tokenizer, text_encoder, device):
    """
    Create object-restricted embeddings (ORE) for each target object
    Returns a list of embeddings: [ore_object1, ore_object2, ..., full_embed]
    """
    ore_embeds = []
    
    # Get substituted base embeddings
    prompt_embeds = get_substituted_base_embeds(
        base_prompt, target_noun_list, tokenizer, text_encoder, device
    )
    
    base_ids = tokenizer(base_prompt).input_ids
    base_EOS_start = len(base_ids) - 1
    
    # Create ORE for each object
    for noun_chunk in target_noun_list:
        prompt_embed = prompt_embeds.clone()
        target_ids = tokenizer(noun_chunk).input_ids[1:-1]
        indices = find_sublist_indices(base_ids, target_ids)
        
        if not indices:
            print(f"Warning: '{noun_chunk}' not found in base prompt")
            continue
        
        target_EOS_start = len(target_ids) + 1
        target_embed = get_single_prompt_embeds(noun_chunk, tokenizer, text_encoder, device)
        
        # Step 1: Zero out all content tokens (keep BOS)
        prompt_embed[0, 1:base_EOS_start] = 0
        
        # Step 2: Keep only the target object tokens
        prompt_embed[0, indices] = target_embed[0, 1:target_EOS_start]
        
        # Step 3: Use target's EOS token (semantic boundary marker)
        prompt_embed[0, base_EOS_start:] = target_embed[
            0, target_EOS_start : 77 - base_EOS_start + target_EOS_start
        ]
        
        ore_embeds.append(prompt_embed)
    
    # Add full prompt embedding at the end
    ore_embeds.append(prompt_embeds)
    
    return ore_embeds

def create_weighted_ore_embedding(ore_embeds, layouts, frame_idx, device):
    """
    Create weighted combination of ORE embeddings based on spatial layouts
    
    Args:
        ore_embeds: List of [1, 77, 768] tensors (one per segment + full)
        layouts: [frames, seg_cls, 1, h, w] spatial layouts
        frame_idx: which frame to process
        device: torch device
    
    Returns:
        weighted_embed: [1, 77, 768] weighted combination
    """
    frames, seg_cls, _, h, w = layouts.shape
    
    # Get layout for this frame: [seg_cls, 1, h, w]
    frame_layout = layouts[frame_idx]  
    
    # Normalize layout to get weights: [seg_cls, 1, h, w]
    # Sum over spatial dimensions to get importance weight for each segment
    segment_weights = frame_layout.sum(dim=[2, 3])  # [seg_cls]
    total_weight = segment_weights.sum()
    
    if total_weight > 0:
        segment_weights = segment_weights / total_weight
    else:
        segment_weights = torch.ones(seg_cls, device=device) / seg_cls
    
    # Create weighted combination
    weighted_embed = torch.zeros_like(ore_embeds[0])
    
    for i in range(seg_cls):
        if i < len(ore_embeds) - 1:  # Exclude the last one (full embed)
            weighted_embed += ore_embeds[i] * segment_weights[i].item()
    
    # Add a small portion of full embedding for global coherence
    weighted_embed += ore_embeds[-1] * 0.1
    
    return weighted_embed
