"""
Analyze why 'Write' methods (like STV activation steering) fail in LMMs.
This tool measures the impact of steering vectors on Visual Attention and Hidden States.
"""
import torch
import numpy as np

def measure_steering_disruption(model_hidden_states_normal, 
                                model_hidden_states_steered, 
                                attention_maps_normal, 
                                attention_maps_steered,
                                image_token_start_idx, 
                                image_token_end_idx):
    """
    Simulated analysis logic to quantify representation disruption.
    
    Args:
        model_hidden_states_normal: Tensor of shape (L, Seq, D)
        model_hidden_states_steered: Tensor of shape (L, Seq, D)
        attention_maps_normal: Attn weights of the final answering token to past tokens
        attention_maps_steered: Attn weights of the final answering token (steered)
        image_token_start_idx, image_token_end_idx: Boundaries of visual tokens
        
    Returns: dict of disruption metrics
    """
    
    # 1. Visual Attention Ratio (How much does the model 'look' at the image?)
    # Sum of attention weights on image tokens divided by total attention
    vis_attn_normal = attention_maps_normal[:, image_token_start_idx:image_token_end_idx].sum().item()
    vis_attn_steered = attention_maps_steered[:, image_token_start_idx:image_token_end_idx].sum().item()
    
    # 2. Representation Corruption (Cosine similarity between normal and steered hidden states)
    # Measured at the final layer, final token
    cos_sim = torch.nn.functional.cosine_similarity(
        model_hidden_states_normal[-1, -1, :], 
        model_hidden_states_steered[-1, -1, :], 
        dim=0
    ).item()
    
    # 3. Vector Norm Mismatch
    # Norm of the difference (the effective steering vector)
    diff = model_hidden_states_steered - model_hidden_states_normal
    steering_norm = torch.norm(diff[-1, -1, :]).item()
    original_norm = torch.norm(model_hidden_states_normal[-1, -1, :]).item()
    
    return {
        "visual_attention_ratio_normal": vis_attn_normal,
        "visual_attention_ratio_steered": vis_attn_steered,
        "attention_drop_percentage": (vis_attn_normal - vis_attn_steered) / (vis_attn_normal + 1e-9) * 100,
        "representation_cosine_similarity": cos_sim,
        "steering_norm_ratio": steering_norm / (original_norm + 1e-9)
    }

# Example wrapper for what you could place in a notebook or script to generate Table/Figure
print("Analysis script ready. You can plug this into your STV or MimIC forward passes.")
print("Expected finding: 'visual_attention_ratio_steered' will be significantly lower, proving the model goes blind.")
