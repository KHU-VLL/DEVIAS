import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from einops import reduce

class TrainLoss(nn.Module):
    """
    https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/models/matcher.py#L12
    """
    def __init__(self,
                 criterion:torch.nn.Module, scene_criterion:torch.nn.Module, num_action_classes:int, slot_matching_method='matching', combine=True, 
                 scene_loss_weight=4000, mask_prediction_loss_weight=1, mask_distill_loss_weight=3):
        super().__init__()
        self.criterion = criterion
        self.scene_criterion = scene_criterion
        self.num_action_classes = num_action_classes
        self.num_scene_classes = 365
        self.slot_matching_method = slot_matching_method
        self.combine = combine
        self.mask_prediction_loss_weight = mask_prediction_loss_weight
        self.mask_distill_loss_weight = mask_distill_loss_weight
        self.scene_loss_weight = scene_loss_weight
        print(f'scene_loss_weight : {self.scene_loss_weight}')
        print(f'mask_prediction_loss_weight : {self.mask_prediction_loss_weight}')
        print(f'mask_distill_loss_weight : {self.mask_distill_loss_weight}')
    
    def forward(self, model, student_output, teacher_outputs, target, fg_mask=None):
        if self.slot_matching_method == 'hard_select':
            (fg_feat, bg_feat), (fg_logit, bg_logit, attn), (slots_head, slots, mask_predictions) = student_output
            bs = target.shape[0]
            device = fg_logit.device
            dtype = fg_logit.dtype
            num_latent = int(slots_head.shape[0] / bs)
            num_head = attn.size(0) // bs
            attn = reduce(attn, '(bs num_head) num_latent dim -> bs num_latent dim', 'mean', num_head=num_head)
            mask_predictions = mask_predictions.reshape(bs,num_latent, -1)

            fg_mask, fg_masks_per_frames = fg_mask
            # fg_mask bs x 196
            # fg_masks_per_frames bs x 1568
            
            slots_head = slots_head.reshape(bs, num_latent, -1)
            
            fg_mask = fg_mask.half() if fg_mask.dtype != torch.float16 else fg_mask
            fg_masks_per_frames = fg_masks_per_frames.half() if fg_masks_per_frames.dtype != torch.float16 else fg_masks_per_frames
            _,teacher_scene_logit = teacher_outputs
            scene_target = torch.argmax(teacher_scene_logit, dim=1)
            var = teacher_scene_logit.min() - float(1)
            inf_tensor = torch.full((scene_target.size(0), self.num_action_classes), var, device=device)
            teacher_scene_logit = torch.cat([inf_tensor, teacher_scene_logit], dim=1)
            action_loss = F.cross_entropy(slots_head[:,0], target)
            scene_loss = F.kl_div(
                F.log_softmax(slots_head[:,1], dim=-1), 
                F.log_softmax(teacher_scene_logit, dim=-1), 
                reduction='batchmean',
                log_target=True
                )  * 4

            mask_distill_loss = F.mse_loss(attn[:,0], fg_masks_per_frames) * self.mask_distill_loss_weight  # distill loss
            mask_prediction_loss = F.binary_cross_entropy_with_logits(  
                mask_predictions[:, 0],
                fg_mask
            ) * self.mask_prediction_loss_weight
            # cosine loss
            slots = slots.reshape(bs,num_latent, -1)
            normed_slots = F.normalize(slots, p=2, dim=2)
            cosine_sim_matrix = torch.bmm(normed_slots, normed_slots.transpose(1, 2))
            identity = torch.eye(cosine_sim_matrix.size(1)).to(cosine_sim_matrix.device)
            cosine_sim_matrix = cosine_sim_matrix * (1 - identity)

            cosine_loss = (cosine_sim_matrix.sum(dim=(1,2)) / (cosine_sim_matrix.size(1) * (cosine_sim_matrix.size(1) - 1))).mean()

            total_loss = action_loss+ scene_loss+mask_distill_loss+mask_prediction_loss +cosine_loss
            return total_loss, \
                fg_logit, \
                {'action_loss':action_loss.item(),
                'scene_loss':scene_loss.item(),
                'mask_distill_loss':mask_distill_loss.item(),
                'mask_prediction_loss':mask_prediction_loss.item(),
                'cosine_loss':cosine_loss.item()}


        elif self.slot_matching_method == 'matching':
            # slot_action_head : (bs x num_slots) x action_classes
            # slots_scene_head : (bs x num_slots) x scene_classes
            (fg_feat, bg_feat), (fg_logit, bg_logit, attn), (slots_head, slots, mask_predictions) = student_output
            device = slots_head.device
            dtype = slots_head.dtype

            bs = target.shape[0]
            num_latent = int(slots_head.shape[0] / bs)
            num_head = attn.size(0) // bs

            # attn mean per head
            attn = reduce(attn, '(bs num_head) num_latent dim -> bs num_latent dim', 'mean', num_head=num_head)
            mask_predictions = mask_predictions.reshape(bs,num_latent, -1)

            _,teacher_scene_logit = teacher_outputs
            scene_target = torch.argmax(teacher_scene_logit, dim=1)
            
            var = teacher_scene_logit.min() - float(1)
            inf_tensor = torch.full((scene_target.size(0), self.num_action_classes), var, device=device)

            teacher_scene_logit = torch.cat([inf_tensor, teacher_scene_logit], dim=1)
            scene_target += self.num_action_classes
            
            slots_head_sfmax = slots_head.softmax(-1)
            slots_action_head_sfmax = slots_head_sfmax[:,:self.num_action_classes]
            slots_scene_head_sfmax = slots_head_sfmax[:,self.num_action_classes:self.num_action_classes+self.num_scene_classes]
            all_indices = []

            for i in range(bs):
                # Compute cost for each query for scene and action for the current image
                cost_action_class = -slots_head_sfmax[i*num_latent:(i+1)*num_latent, target[i]]
                cost_scene_class = -slots_head_sfmax[i*num_latent:(i+1)*num_latent, scene_target[i]]
                
                # Concatenate the two costs
                combined_cost = torch.cat([cost_action_class.unsqueeze(-1), cost_scene_class.unsqueeze(-1)], dim=1)
                
                # Use Hungarian algorithm on the combined cost
                indices = linear_sum_assignment(combined_cost.detach().cpu())
                all_indices.append(indices)

            # Convert the list of indices into the desired format
            all_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in all_indices]

            action_loss = torch.tensor([0],device=device, dtype=dtype)
            scene_loss = torch.tensor([0],device=device, dtype=dtype)
            selected_slots = []
            action_logit = []
            slots_head = slots_head.view(bs,num_latent,-1)
            fg_mask, fg_masks_per_frames = fg_mask
            # fg_mask bs x 196
            # fg_masks_per_frames bs x 1568
            
            fg_mask = fg_mask.half() if fg_mask.dtype != torch.float16 else fg_mask
            fg_masks_per_frames = fg_masks_per_frames.half() if fg_masks_per_frames.dtype != torch.float16 else fg_masks_per_frames
            
            mask_prediction_loss = torch.tensor([0],device=device, dtype=dtype)
            mask_distill_loss = torch.tensor([0],device=device, dtype=dtype)
            
            for batch_idx, (slot_indices, label_indices) in enumerate(all_indices):
                for s_idx, l_idx in zip(slot_indices, label_indices):
                    if l_idx == 0:  # action
                        mask_distill_loss += F.mse_loss(attn[batch_idx, s_idx], fg_masks_per_frames[batch_idx]) * self.mask_distill_loss_weight  # distill loss
                        mask_prediction_loss += F.binary_cross_entropy_with_logits(  
                            mask_predictions[batch_idx, s_idx],
                            fg_mask[batch_idx]
                        ) * self.mask_prediction_loss_weight
                        action_loss += F.cross_entropy(slots_head[batch_idx, s_idx], target[batch_idx])
                        action_logit.append(slots_head[batch_idx, s_idx])
                        selected_slots.append((batch_idx, int(s_idx)))
                    elif l_idx == 1:  # scene
                        if self.scene_criterion == "CE":
                            scene_loss += F.cross_entropy(slots_head[batch_idx, s_idx], scene_target[batch_idx])
                        
                        elif self.scene_criterion == "KL":
                            # kl diviergence
                            scene_loss += F.kl_div(
                                F.log_softmax(slots_head[batch_idx, s_idx], dim=-1), 
                                F.log_softmax(teacher_scene_logit[batch_idx], dim=-1), 
                                reduction='batchmean',
                                log_target=True
                                ) * self.scene_loss_weight
                        selected_slots.append((batch_idx, int(s_idx)))

            action_loss /= bs
            scene_loss /= bs
            mask_prediction_loss /= bs
            mask_distill_loss /= bs
                
            slots = slots.reshape(bs,num_latent, -1)
            normed_slots = F.normalize(slots, p=2, dim=2)
            cosine_sim_matrix = torch.bmm(normed_slots, normed_slots.transpose(1, 2))
            identity = torch.eye(cosine_sim_matrix.size(1)).to(cosine_sim_matrix.device)
            cosine_sim_matrix = cosine_sim_matrix * (1 - identity)
            cosine_loss = (cosine_sim_matrix.sum(dim=(1,2)) / (cosine_sim_matrix.size(1) * (cosine_sim_matrix.size(1) - 1))).mean()


            total_loss = action_loss + scene_loss + cosine_loss + mask_prediction_loss + mask_distill_loss
            return total_loss, \
                torch.stack(action_logit), \
                {'action_loss':action_loss.item(),
                'scene_loss':scene_loss.item(),
                'cosine_loss':cosine_loss.item(),
                'mask_prediction_loss':mask_prediction_loss.item(),
                'mask_distill_loss':mask_distill_loss.item()}