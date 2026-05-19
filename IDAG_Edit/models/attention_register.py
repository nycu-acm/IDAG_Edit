"""
Register attention editor to Diffuser Pipeline, refer from [https://github.com/google/prompt-to-prompt]
"""
import torch
import torch.nn.functional as F
from einops import rearrange

def regiter_crossattn_editor_diffusers_p2p(model, controller): 
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None): 
                is_cross = encoder_hidden_states is not None
                assert is_cross
                # batch_size, sequence_length, _ = hidden_states.shape # [bdz: 64, 4096, 320]
                encoder_hidden_states = encoder_hidden_states   # [64, seq_len:77, dim:768]
                encoder_hidden_states = controller.text_cond_current if encoder_hidden_states is not None else encoder_hidden_states
                
                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)

                if self.added_kv_proj_dim is not None:
                    key = self.to_k(hidden_states)
                    value = self.to_v(hidden_states)
                    encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                    encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                    ######record###### record before reshape heads to batch dim
                    if self.processor is not None:
                        self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
                    ##################

                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)
                    encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                    encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                    key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                    value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
                else:
                    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                    key = self.to_k(encoder_hidden_states)
                    value = self.to_v(encoder_hidden_states)

                    if self.processor is not None:
                        self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)

                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)

                query = self.reshape_heads_to_batch_dim(query) # reshape query

                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                        attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

                if self.processor is not None:
                    self.processor.record_attn_mask(self, hidden_states, query, key, value, attention_mask)

                # query shape     [512, spatia_size, head_dim(40/80/160)]
                # key-value shape [512, 77 head_dim(40/80/160)]
                
                ######start of layout control######
                if is_cross: 
                    sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale    # QKT/d
                    if hasattr(controller, 'get_layout_bias'):
                        attention_probs = controller.get_layout_bias(sim, is_cross, place_in_unet)
                        if attention_probs is not None:       # not all of steps use layout bias (ex:15/50 steps)
                            sim = attention_probs

                        ### 4d tensor version ###
                        # attention_probs = controller.get_layout_bias(reshape_batch_dim_to_temporal_heads(sim), is_cross, place_in_unet)
                        # if attention_probs is not None:
                        #     sim = reshape_temporal_heads_to_batch_dim(attention_probs)
                    
                    # attention, what we cannot get enough of
                    attn = sim.softmax(dim=-1)
                    attn = controller(attn, is_cross, place_in_unet)    # AttentionControlEdit
                    hidden_states = torch.einsum("b i j, b j d -> b i d", attn, value)
                else:
                    assert is_cross, "shouldn't be not-cross!"
                #######End of layout control######
                hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
                
                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states
                ######

        def reshape_temporal_heads_to_batch_dim(tensor):
            head_size = self.heads
            tensor = rearrange(tensor, " b h s t -> (b h) s t ", h = head_size)
            return tensor

        def reshape_batch_dim_to_temporal_heads(tensor):
            head_size = self.heads
            tensor = rearrange(tensor, "(b h) s t -> b h s t", h = head_size)
            return tensor
        
        return forward
    
        
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention': # or net_.__class__.__name__ == 'SelfAttention': # or net_.__class__.__name__ == 'VanillaTemporalModule':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            cross_att_count += register_recr(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_recr(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_recr(net, 0, "up")
    controller.num_att_layers = cross_att_count
    print("p2p_cross_att_count: ", controller.num_att_layers)
    
 
def regiter_selfattn_editor_diffusers_p2p(model, editor): 
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
                is_cross = encoder_hidden_states is not None
                assert not is_cross, "shouldn't be cross!"
                batch_size, sequence_length, _ = hidden_states.shape
                encoder_hidden_states = encoder_hidden_states

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                dim = query.shape[-1]

                if self.added_kv_proj_dim is not None:
                    key = self.to_k(hidden_states)
                    value = self.to_v(hidden_states)
                    encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                    encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                    ######record###### record before reshape heads to batch dim
                    if self.processor is not None:
                        self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)
                    encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                    encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                    key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                    value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
                else:
                    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                    key = self.to_k(encoder_hidden_states)
                    value = self.to_v(encoder_hidden_states)
                    if self.processor is not None:
                        self.processor.record_qkv(self, hidden_states, query, key, value, attention_mask)
                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)

                query = self.reshape_heads_to_batch_dim(query) # reshape query

                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                        attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
                
                if self.processor is not None:
                    self.processor.record_attn_mask(self, hidden_states, query, key, value, attention_mask)
                hidden_states = editor(q=query, k=key, v=value, attention_mask=attention_mask, batch_size=batch_size, num_heads=self.heads, scale=self.scale)
                # linear proj
                hidden_states = self.to_out[0](hidden_states)

                # dropout
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states
                ##########################################
        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'SelfAttention': # net_.__class__.__name__ == 'CrossAttention' or  # or net_.__class__.__name__ == 'VanillaTemporalModule':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    self_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            self_att_count += register_recr(net, 0, "down")
        elif "mid" in net_name:
            self_att_count += register_recr(net, 0, "mid")
        elif "up" in net_name:
            self_att_count += register_recr(net, 0, "up")
    editor.num_att_layers = self_att_count
    print("self_att_count: ", editor.num_att_layers)
    
  