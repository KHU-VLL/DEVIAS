import random
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips
import torch
import torch.nn as nn
import kornia
import torchvision.transforms as transforms
import numpy as np
from kornia.augmentation.container import VideoSequential
import torch.nn.functional as F


class FAME(nn.Module):
    def __init__(self, crop_size=112, beta=0.5, device="cpu", eps=1e-8,prob_aug=0.5):
        super(FAME, self).__init__()
        self.frame_mean=[0.485, 0.456, 0.406],
        self.frame_std=[0.229, 0.224, 0.225]
        self.crop_size = crop_size
        gauss_size = int(0.1 * crop_size) // 2 * 2 + 1
        self.gauss = kornia.filters.GaussianBlur2d(
            (gauss_size, gauss_size),
            (gauss_size / 3, gauss_size / 3))
        self.device = device
        self.eps = eps
        self.beta = beta # control the portion of foreground
        self.prob_aug = prob_aug

    #### min-max normalization
    def norm_batch(self, matrix):
        # matrix : B*H*W
        B, H, W = matrix.shape
        matrix = matrix.flatten(start_dim=1)
        matrix -= matrix.min(dim=-1, keepdim=True)[0]
        matrix /= (matrix.max(dim=-1, keepdim=True)[0] + self.eps)
        return matrix.reshape(B, H, W)

    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        values = torch.ones_like(x)
        target.scatter_add_(dim, x, values)
        return target
    
    def getSeg(self, mask, video_clips):
        # input mask:B, H, W; video_clips:B, C, T, H, W
        # return soft seg mask: B, H, W
        B, C, T, H, W = video_clips.shape
        video_clips_ = video_clips.mean(dim=2) # B, C, H, W
        img_hsv = kornia.color.rgb_to_hsv(video_clips_.reshape(-1, C, H, W))  # B, C, H, W
        sampled_fg_index = torch.topk(mask.reshape(B, -1), k=int(0.5 * H * W), dim=-1)[1]  # shape B * K
        sampled_bg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * H * W), dim=-1, largest=False)[1]  # shape B * K
        
        dimH, dimS, dimV = 10, 10, 10
        img_hsv = img_hsv.reshape(B, -1, H, W)  # B * C * H * W
        img_h = img_hsv[:, 0]
        img_s = img_hsv[:, 1]
        img_v = img_hsv[:, 2]
        hx = (img_s * torch.cos(img_h * 2 * np.pi) + 1) / 2
        hy = (img_s * torch.sin(img_h * 2 * np.pi) + 1) / 2
        h = torch.round(hx * (dimH - 1) + 1)
        s = torch.round(hy * (dimS - 1) + 1)
        v = torch.round(img_v * (dimV - 1) + 1)
        color_map = h + (s - 1) * dimH + (v - 1) * dimH * dimS  # B, H, W
        color_map = color_map.reshape(B, -1).long()
        col_fg = color_map.gather(index=sampled_fg_index, dim=-1)  # B * K
        col_bg = color_map.gather(index=sampled_bg_index, dim=-1)  # B * K
        dict_fg = self.batched_bincount(col_fg, dim=1, max_value=dimH * dimS * dimV)  # B * (dimH * dimS * dimV)
        dict_bg = self.batched_bincount(col_bg, dim=1, max_value=dimH * dimS * dimV)  # B * (dimH * dimS * dimV)
        dict_fg = dict_fg.float()
        dict_bg = dict_bg.float() + 1
        dict_fg /= (dict_fg.sum(dim=-1, keepdim=True) + self.eps)
        dict_bg /= (dict_bg.sum(dim=-1, keepdim=True) + self.eps)

        pr_fg = dict_fg.gather(dim=1, index=color_map)
        pr_bg = dict_bg.gather(dim=1, index=color_map)
        refine_mask = pr_fg / (pr_bg + pr_fg)

        mask = self.gauss(refine_mask.reshape(-1, 1, H, W))
        mask = self.norm_batch(mask.reshape(-1, H, W))
        
        num_fg = int(self.beta * H * W)
        sampled_index = torch.topk(mask.reshape(B, -1), k=num_fg, dim=-1)[1]
        mask = torch.zeros_like(mask).reshape(B, -1)
        b_index = torch.LongTensor([[i]*num_fg for i in range(B)])
        mask[b_index.view(-1), sampled_index.view(-1)] = 1
        return mask.reshape(B, H, W)

    def getmask(self, video_clips):
        # input video_clips: B, C, T, H, W
        # return soft seg mask: B, H, W
        B, C, T, H, W = video_clips.shape
        im_diff = (video_clips[:, :, 0:-1] - video_clips[:, :, 1:]).abs().sum(dim=1).mean(dim=1)  # B, H, W
        mask = self.gauss(im_diff.reshape(-1, 1, H, W))
        mask = self.norm_batch(mask.reshape(-1, H, W))  # B, H, W
        mask = self.getSeg(mask, video_clips)
        return mask

    def getmask_per_frame(self, video_clips):
        # input frame: B, C, H, W
        # return soft seg mask: B, H, W
        B, C, T, H, W = video_clips.shape
        masks = []
        for i in range(0, T, 2):
            im_diff = (video_clips[:, :, i] - video_clips[:, :, i+1]).abs().sum(dim=1)  # B, H, W
            mask = self.gauss(im_diff.reshape(-1, 1, H, W))
            mask = self.norm_batch(mask.reshape(-1, H, W))  # B, H, W
            mask = self.getSeg(mask, video_clips)
            masks.append(mask)
        return masks

    def forward(self, videos, label, center_frame=None):
        indicator = len(videos.shape)
        batch_size, channel, num_clip, h, w = videos.shape
        tmp_video = videos.contiguous()

        tmp_video = tmp_video * torch.tensor(self.frame_std, device=tmp_video.device).reshape(1,3,1,1,1) + torch.tensor(self.frame_mean, device=tmp_video.device).reshape(1,3,1,1,1) #denormalized
        mask = self.getmask(tmp_video) #B, H, W
        masks_per_frame = self.getmask_per_frame(tmp_video) #B,T, H, W
        masks_per_frame = torch.stack(masks_per_frame).permute(1, 0, 2, 3)
        mask=mask.to(videos.dtype)
        masks_per_frame=masks_per_frame.to(videos.dtype)
        mask = mask.unsqueeze(1).unsqueeze(1)  #! 3, 224, 224 -> 3, 1, 1, 224, 224
        index = torch.randperm(batch_size, device=videos.device)
        video_fuse = videos[index] * (1 - mask) + videos * mask

        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_videos = torch.cat([video_fuse[aug_ind], videos[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([label[aug_ind], label[ori_ind]], dim=0).contiguous()
            if center_frame is not None :
                all_center_frame = torch.cat([center_frame[aug_ind], center_frame[ori_ind]], dim=0).contiguous()
            mask = torch.cat([mask[aug_ind], mask[ori_ind]], dim=0).contiguous()
            masks_per_frame = torch.cat([masks_per_frame[aug_ind], masks_per_frame[ori_ind]], dim=0).contiguous()
        else:
            all_videos = video_fuse
            all_label = label
            if center_frame is not None :
                all_center_frame = center_frame
                
        pooled_data = F.avg_pool2d(mask.squeeze(1).squeeze(1), kernel_size=16, stride=16)
        reshaped_data = pooled_data.view(batch_size, -1)
        mask = reshaped_data.to(label.device, non_blocking=True)
        
        pooled_data = F.avg_pool2d(masks_per_frame.squeeze(2), kernel_size=16, stride=16)
        reshaped_data = pooled_data.view(batch_size, -1)
        masks_per_frame = reshaped_data.to(label.device, non_blocking=True)
        
        if center_frame is not None :
            return all_videos, all_label,(mask,masks_per_frame), all_center_frame
        else :
            return all_videos, all_label,(mask,masks_per_frame)
            
if __name__ == "__main__":
    from decord import VideoReader
    from decord import cpu, gpu
    import matplotlib.pyplot as plt

    fame = FAME()
    samples = torch.rand(8,3,16,224,224).cuda()
    targets = torch.rand(8).cuda()
    fame(samples,targets)