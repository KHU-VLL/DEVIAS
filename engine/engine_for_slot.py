import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from utils.transform.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils.utils as utils
from scipy.special import softmax
import torch.nn.functional as F
from torch import nn
from einops import rearrange


def segformer_mix_sample(mask , videos,label,args):
    # mask B T H W   .. T = 8
    mask = mask.to(videos.dtype)
    batch_size, channel, num_clip, h, w = videos.shape
    tmp_video = videos.contiguous()
    masks_per_frame = torch.repeat_interleave(mask, repeats=2, dim=1) # mask B T H W   .. T = 16
    index = torch.randperm(batch_size, device=videos.device)
    # video_fuse = videos[index] * (1 - mask) + videos * mask # mix by single mask
    video_fuse = videos[index] * (1 - masks_per_frame.unsqueeze(1)) + videos * masks_per_frame.unsqueeze(1)

    ## choose samples according to prob
    if args.prob_aug < 1:
        rand_batch = torch.rand(batch_size)
        aug_ind = torch.where(rand_batch < args.prob_aug)
        ori_ind = torch.where(rand_batch >= args.prob_aug)
        all_videos = torch.cat([video_fuse[aug_ind], videos[ori_ind]], dim=0).contiguous()
        all_label = torch.cat([label[aug_ind], label[ori_ind]], dim=0).contiguous()
        all_mask = torch.cat([mask[aug_ind], mask[ori_ind]], dim=0).contiguous()

    else:
        all_videos = video_fuse
        all_label = label
        all_mask = mask
        
    pooled_data = F.avg_pool2d(all_mask, kernel_size=16, stride=16)
    random_index = torch.randint(0, 8, (1,))
    random_mask = pooled_data[:, random_index, :, :].view(batch_size, -1)

    
    reshaped_data = pooled_data.view(batch_size, -1)
    masks_per_frame = reshaped_data.to(label.device, non_blocking=True) # bs x 1568
    return all_videos, all_label,(random_mask,masks_per_frame)


def train_class_batch(model, scene_model,samples, target, train_criterion, fg_mask=None):
    student_output = model(samples)
    with torch.no_grad():
        teacher_output = scene_model(samples,return_attn=False)
        
    total_loss, output, loss_dict = train_criterion(model, student_output,teacher_output, target, fg_mask=fg_mask)
    return total_loss,output,loss_dict


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,scene_model: torch.nn.Module, train_criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, mask_model=None,args=None):
    model.train(True)
    scene_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            if 'FAME' in str(mask_model):
                samples, targets, masks = mask_model(samples, targets) # mask (bs, 1, 1, H, W)
                samples = samples.half()
                
            elif 'Segformer' in str(mask_model):
                samples = samples.half()
                with torch.no_grad():
                    num_frames = samples.shape[2] // 2
                    fg_masks = mask_model.forward(samples[:,:,::2].squeeze(0).permute(0,2,1,3,4).reshape((samples.shape[0]*num_frames,samples.shape[1],samples.shape[3],samples.shape[4])))
                    mask=F.interpolate(fg_masks.logits, scale_factor=4, mode='bilinear', align_corners=False)
                    mask = mask.max(dim=1)[1] == 11
                    mask = mask.reshape(targets.size(0),num_frames,-1,samples.size(-1))
                    samples,targets,masks = segformer_mix_sample(mask , samples,targets,args)

            loss, output,loss_dict = train_class_batch(
                    model,scene_model, samples, targets, train_criterion, fg_mask=masks)

        else:
            with torch.cuda.amp.autocast():
                ### MASK
                if 'FAME' in str(mask_model):
                    samples, targets, masks = mask_model(samples, targets) # mask (bs,1,1,H,W)
                elif 'Segformer' in str(mask_model):
                    with torch.no_grad():
                        num_frames = samples.shape[2] // 2
                        fg_masks = mask_model.forward(samples[:,:,::2].squeeze(0).permute(0,2,1,3,4).reshape((samples.shape[0]*num_frames,samples.shape[1],samples.shape[3],samples.shape[4])))
                        mask=F.interpolate(fg_masks.logits, scale_factor=4, mode='bilinear', align_corners=False)
                        mask = mask.max(dim=1)[1] == 11
                        mask = mask.reshape(targets.size(0),num_frames,-1,samples.size(-1))
                        samples,targets,masks = segformer_mix_sample(mask , samples,targets,args)

                loss, output,loss_dict = train_class_batch(
                    model,scene_model, samples, targets, train_criterion, fg_mask=masks)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
            
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v})
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            for k, v in loss_dict.items():
                log_writer.update(**{k: v},head='loss')
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device,args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 50, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = videos.shape[0]

        # compute output
        with torch.cuda.amp.autocast():
            _, (output, scene_output, attn), _ = model(videos)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5,
                multi_slot_acc=metric_logger.multi_slot_acc, 
                losses=metric_logger.loss,
                ))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 100, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            _, (output, scene_output, attn), _ = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test_with_scene_label(data_loader, model, scene_model, device, file, num_labels=400):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    scene_model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 100, header):
        videos = batch[0]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            _, (action_output, output, attn), _ = model(videos)
            output = output[:, num_labels:]  #! for unified head
            
            #! get scene label from teacher
            with torch.no_grad():
                _, teacher_scene_logit = scene_model(videos, return_attn=False)
            target = torch.argmax(teacher_scene_logit, dim=1)

            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]