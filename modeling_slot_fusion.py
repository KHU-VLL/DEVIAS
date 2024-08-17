from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
from agg_block.agg_block import AggregationBlock



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=[768, 768], fc_drop_rate=0.):
        super(MLPHead, self).__init__()
        print(hidden_sizes[0])
        self.fc_fg_down = nn.Linear(in_dim, in_dim // 2)
        self.fc_bg_down = nn.Linear(in_dim, in_dim // 2)
        
        self.fc_fg_ln = nn.LayerNorm(in_dim // 2)
        self.fc_bg_ln = nn.LayerNorm(in_dim // 2)
        
        self.fc_input_ln = nn.LayerNorm(in_dim)

        self.classifier = nn.Linear(in_dim, out_dim)
        
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.relu = nn.ReLU()

        
    def forward(self, fg_token, bg_token):
        fg_token = self.fc_fg_ln(self.fc_fg_down(fg_token))
        bg_token = self.fc_fg_ln(self.fc_fg_down(bg_token))
        
        output = torch.concat([fg_token, bg_token], dim=1)
        
        output = self.classifier(self.fc_dropout(self.relu(self.fc_input_ln(output))))

        return output

class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Only drop paths for the batch dimension
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize
            output = x.div(keep_prob) * random_tensor
            return output
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x,return_attn=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x,return_attn=True):
        if return_attn:
            new_x,attn = self.attn(self.norm1(x),return_attn)
            x = x + self.drop_path(new_x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn

        else:
            x = x + self.drop_path(self.attn(self.norm1(x),return_attn))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 


class MaskPredictor(nn.Module):
    def __init__(self):
        super(MaskPredictor, self).__init__()
        
        self.conv1 = nn.Conv2d(770, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        
        self.decoder = nn.Sequential(
            nn.Linear(768, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 196),
        )
        
        self.act = nn.ReLU()
        
    def forward(self, cls_token):
        
        width = 14
        
        mask = self.decoder(cls_token)
        
        return mask.squeeze().reshape(cls_token. shape[0], width*width)  # Remove extra dimension for [batch, 14, 14]


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True,
                 num_latents=4,
                 residual=False,
                 head_type='linear',
                 fusion='hard_select',
                 use_adapter=False,
                 key_softmax=False,
                 disentangle_criterion=None,
                 no_label=False,
                 weight_tie_layers=True,
                 agg_depth=4,
                 num_scene_classes=365,
                 slot_fusion='concat',
                 downstream_num_classes=50
                 ):
        super().__init__()
        self.num_slots = num_latents
        self.num_classes = num_classes
        self.num_scene_classes = num_scene_classes
        
        self.residual = residual
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.disentangle_criterion = disentangle_criterion
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        if not use_mean_pooling:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
            num_patches += 1
        self.slot_fusion = slot_fusion
        self.fusion = fusion

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()

        self.agg_block = AggregationBlock(num_latents=num_latents, key_softmax=key_softmax, 
                                          weight_tie_layers=weight_tie_layers, depth=agg_depth)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.no_label = no_label

        self.fg_norm =  norm_layer(embed_dim)
        self.bg_norm =  norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes+self.num_scene_classes)
        if head_type == 'linear':
            if self.slot_fusion=='concat':
                self.fusion_head = nn.Linear(embed_dim * num_latents, downstream_num_classes) if downstream_num_classes > 0 else nn.Identity()
            elif self.slot_fusion =='sum':
                self.fusion_head = nn.Linear(embed_dim, downstream_num_classes) if downstream_num_classes > 0 else nn.Identity()
            elif self.slot_fusion =='fg_only' or self.slot_fusion =='bg_only':
                self.fusion_head = nn.Linear(embed_dim, downstream_num_classes) if downstream_num_classes > 0 else nn.Identity()
            elif self.slot_fusion =='backbone':
                self.fusion_head = nn.Linear(embed_dim, downstream_num_classes) if downstream_num_classes > 0 else nn.Identity()      
            trunc_normal_(self.fusion_head.weight, std=.02)
            self.apply(self._init_weights)
            self.fusion_head.weight.data.mul_(init_scale)
            self.fusion_head.bias.data.mul_(init_scale)
        else:
            #mlp
            if self.slot_fusion == 'concat':
                self.fusion_head = MLPHead(embed_dim, downstream_num_classes, fc_drop_rate=fc_drop_rate) if downstream_num_classes > 0 else nn.Identity()
            elif self.slot_fusion == 'sum':
                self.fusion_head = MLPHead(embed_dim, downstream_num_classes,hidden_sizes=[768*2,768*2]) if downstream_num_classes > 0 else nn.Identity()
            elif self.slot_fusion =='fg_only' or self.slot_fusion =='bg_only':
                self.fusion_head = MLPHead(embed_dim, downstream_num_classes,hidden_sizes=[768*2,768*2]) if downstream_num_classes > 0 else nn.Identity()

            trunc_normal_(self.fusion_head.classifier.weight, std=.02)
            
            self.apply(self._init_weights)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x,return_attn=False):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        attn_list = []

        if self.use_checkpoint:
            for blk in self.blocks:
                x= checkpoint.checkpoint(blk, x)
                # attn_list.append(attn)
        else:  
            if return_attn: 
                for blk in self.blocks:
                    x, attn = blk(x,return_attn=return_attn)
                    attn_list.append(attn)
            else:
                for blk in self.blocks:
                    x = blk(x,return_attn=return_attn)
                
        x = self.norm(x)
        return x

    def forward(self, x,return_attn=False):
        x = self.forward_features(x,return_attn)
        if self.slot_fusion == 'backbone':
            x = self.fc_dropout(self.fg_norm(x.mean(1)))
            x = self.fusion_head(x)
            return [],x

        #TODO token merging
        slots, attn  = self.agg_block(x)  


        bs, num_slots, _ = slots.size()
        slots = slots.reshape(-1, 768)
        slots_head = self.head(slots)
        
        slot_probs = F.softmax(slots_head, dim=-1).view(bs, num_slots, -1)

        action_softmax_output = slot_probs[:,:, :self.num_classes]
        scene_softmax_output = slot_probs[:,:, self.num_classes:self.num_classes+self.num_scene_classes]
        
        action_max_slot_indices = torch.argmax(action_softmax_output.max(dim=-1).values, dim=1)
        scene_max_slot_indices = torch.argmax(scene_softmax_output.max(dim=-1).values, dim=1)

        fg_feat = slots.view(bs, num_slots, -1)[torch.arange(bs), action_max_slot_indices]
        bg_feat = slots.view(bs, num_slots, -1)[torch.arange(bs), scene_max_slot_indices]
        # fg_logit = slots_head.view(bs, num_slots, -1)[torch.arange(bs), action_max_slot_indices]
        # bg_logit = slots_head.view(bs, num_slots, -1)[torch.arange(bs), scene_max_slot_indices]
        
        # return (fg_feat,bg_feat), (fg_logit,bg_logit,attn),(slots_head,slots,mask_predictions)
        
        fg_feat = self.fg_norm(fg_feat)
        bg_feat = self.bg_norm(bg_feat)

        if self.slot_fusion == 'concat':
            input = torch.concat((fg_feat, bg_feat), dim=1)
            output = self.fusion_head(fg_feat, bg_feat)
            
            return input, output
        
        elif self.slot_fusion == 'sum':
            input = fg_feat + bg_feat
            output = self.fusion_head(input)
            return input,output
        
        elif self.slot_fusion == 'fg_only':
            input = fg_feat
            output = self.fusion_head(input)
            return input,output
        
        elif self.slot_fusion == 'bg_only':
            input = bg_feat
            output = self.fusion_head(input)
            return input,output
        
        else:
            raise ValueError('fusion error')


@register_model
def slot_fusion_vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()    
    return model