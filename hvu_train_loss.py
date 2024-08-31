import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from einops import reduce
from run_slot_finetuning_hvu import HVU_NUM_ACTION_CLASSES, HVU_NUM_SCENE_CLASSES

class TrainLoss(nn.Module):
    """
    https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/models/matcher.py#L12
    """
    def __init__(self,
                 criterion:torch.nn.Module, scene_criterion:torch.nn.Module, slot_matching_method='matching', mask_prediction_loss_weight=1.0, mask_distill_loss_weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.scene_criterion = scene_criterion
        self.num_action_classes = HVU_NUM_ACTION_CLASSES   #! for hvu
        self.num_scene_classes = HVU_NUM_SCENE_CLASSES
        self.slot_matching_method = slot_matching_method
        self.mask_prediction_loss_weight = mask_prediction_loss_weight
        self.mask_distill_loss_weight = mask_distill_loss_weight

        print(f'scene_criterion : {self.scene_criterion}')
        print(f'mask_prediction_loss_weight : {self.mask_prediction_loss_weight}')
        print(f'mask_distill_loss_weight : {self.mask_distill_loss_weight}')

    def forward(self, student_output, action_targets, scene_targets, fg_mask=None):
        #! implemented only for matching
        if self.slot_matching_method == 'matching' :
            # slot_action_head : (bs x num_slots) x action_classes
            # slots_scene_head : (bs x num_slots) x scene_classes
            _, (_, _, attn), (slots_head, slots, mask_predictions) = student_output
            device = slots_head.device
            dtype = slots_head.dtype

            bs = action_targets.shape[0]
            target = action_targets
            num_latent = int(slots_head.shape[0] / bs)
            num_head = attn.size(0) // bs

            # attn mean per head
            attn = reduce(attn, '(bs num_head) num_latent dim -> bs num_latent dim', 'mean', num_head=num_head)
            mask_predictions = mask_predictions.reshape(bs, num_latent, -1)

            scene_target = scene_targets
            scene_target += self.num_action_classes
            
            slots_head_sfmax = slots_head.softmax(-1)
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
            # fg_mask (bs x 196)
            # fg_masks_per_frames (bs x 1568)
            
            fg_mask = fg_mask.half() if fg_mask.dtype != torch.float16 else fg_mask
            fg_masks_per_frames = fg_masks_per_frames.half() if fg_masks_per_frames.dtype != torch.float16 else fg_masks_per_frames
            
            mask_prediction_loss = torch.tensor([0],device=device, dtype=dtype)
            mask_distill_loss = torch.tensor([0],device=device, dtype=dtype)
            
            for batch_idx, (slot_indices, label_indices) in enumerate(all_indices):
                for s_idx, l_idx in zip(slot_indices, label_indices):
                    if l_idx == 0:  # action
                        mask_distill_loss += F.mse_loss(attn[batch_idx,s_idx], fg_masks_per_frames[batch_idx]) * self.mask_distill_loss_weight  # distill loss
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
                            logit = slots_head[batch_idx, s_idx]
                            target_index = scene_target[batch_idx]
                            log_prob = F.log_softmax(logit.unsqueeze(0), dim=1)  
                            _scene_target = torch.zeros_like(log_prob).scatter_(1, target_index.view(1, 1), 1)
                            scene_loss += F.kl_div(log_prob, _scene_target, reduction='batchmean')
                            
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
                torch.stack(action_logit),\
                {'action_loss':action_loss.item(),
                'scene_loss':scene_loss.item(),
                'cosine_loss':cosine_loss.item(),
                'mask_prediction_loss':mask_prediction_loss.item(),
                'mask_distill_loss':mask_distill_loss.item()}
        else :
            raise NotImplementedError()