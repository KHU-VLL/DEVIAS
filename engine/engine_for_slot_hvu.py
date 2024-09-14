import math
import sys
from typing import Iterable, Optional
import torch
from utils.transform.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils.utils as utils

from run_slot_finetuning_hvu import HVU_NUM_ACTION_CLASSES


def train_class_batch(model,samples, action_targets, scene_targets, train_criterion, fg_mask=None):
    student_output = model(samples)
    total_loss, output, loss_dict = train_criterion(student_output, action_targets, scene_targets, fg_mask=fg_mask)
    return total_loss,output,loss_dict


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, train_criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, mask_model=None,args=None):
    model.train(True)
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

    for data_iter_step, (samples, action_targets, scene_targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
        action_targets = action_targets.to(device, non_blocking=True)
        scene_targets = scene_targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            if 'FAME' in str(mask_model):
                samples, action_targets, scene_targets, masks = mask_model(samples, action_targets, scene_targets) # mask (bs, 1, 1, H, W)
                samples = samples.half()
                
            loss, output,loss_dict = train_class_batch(
                    model, samples, action_targets,scene_targets,train_criterion, fg_mask=masks)

        else:
            with torch.cuda.amp.autocast():
                ### MASK
                if 'FAME' in str(mask_model):
                    samples, action_targets, scene_targets, masks = mask_model(samples, action_targets, scene_targets) # mask (bs, 1, 1, H, W)
                loss, output,loss_dict = train_class_batch(
                    model, samples, targets, train_criterion, fg_mask=masks)

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
            class_acc = (output.max(-1)[-1] == action_targets).float().mean()
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
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 50, header):
        videos = batch[0]
        action_targets = batch[1]
        scene_targets = batch[2]
        videos = videos.to(device, non_blocking=True)

        action_targets = action_targets.to(device, non_blocking=True)
        scene_targets = scene_targets.to(device, non_blocking=True)
        batch_size = videos.shape[0]
        #! for unified_head
        scene_targets += HVU_NUM_ACTION_CLASSES

        # compute output
        with torch.cuda.amp.autocast():
            _, (action_output, scene_output, attn), _ = model(videos)
            loss = criterion(action_output, action_targets)

        action_acc1, action_acc5 = accuracy(action_output, action_targets, topk=(1, 5))
        scene_acc1, scene_acc5 = accuracy(scene_output, scene_targets, topk=(1, 5))
        
        metric_logger.update(loss=loss.item())
        metric_logger.meters['action_acc1'].update(action_acc1.item(), n=batch_size)
        metric_logger.meters['action_acc5'].update(action_acc5.item(), n=batch_size)

        metric_logger.meters['scene_acc1'].update(scene_acc1.item(), n=batch_size)
        metric_logger.meters['scene_acc5'].update(scene_acc5.item(), n=batch_size)
  
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Action Acc@1 {action_top1.global_avg:.3f} Action Acc@5 {action_top5.global_avg:.3f} | Scene Acc@1 {scene_top1.global_avg:.3f} Scene Acc@5 {scene_top5.global_avg:.3f}  | loss {losses.global_avg:.3f}'
          .format(action_top1=metric_logger.action_acc1, action_top5=metric_logger.action_acc5,scene_top1=metric_logger.scene_acc1, scene_top5=metric_logger.scene_acc5, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_action(data_loader, model, device, header='Val:'):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = header

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 50, header):
        videos = batch[0]
        action_targets = batch[1]
        scene_targets = batch[2]
        action_targets = action_targets.to(device, non_blocking=True)
        scene_targets = scene_targets.to(device, non_blocking=True)
        videos = videos.to(device, non_blocking=True)
        scene_targets += HVU_NUM_ACTION_CLASSES

        # compute output
        with torch.cuda.amp.autocast():
            _, (action_output, scene_output, attn), _ = model(videos)
            loss = criterion(action_output, action_targets)

        action_acc1, action_acc5 = accuracy(action_output, action_targets, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['action_acc1'].update(action_acc1.item(), n=batch_size)
        metric_logger.meters['action_acc5'].update(action_acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Action Acc@1 {action_top1.global_avg:.3f} Action Acc@5 {action_top5.global_avg:.3f} | loss {losses.global_avg:.3f}'
          .format(action_top1=metric_logger.action_acc1, action_top5=metric_logger.action_acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def validation_scene(data_loader, model, device, header='Val:'):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = header

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 50, header):
        videos = batch[0]
        action_targets = batch[1]
        scene_targets = batch[2]
        action_targets = action_targets.to(device, non_blocking=True)
        scene_targets = scene_targets.to(device, non_blocking=True)
        videos = videos.to(device, non_blocking=True)
        #! for unified_head
        scene_targets += HVU_NUM_ACTION_CLASSES

        # compute output
        with torch.cuda.amp.autocast():
            _, (action_output, scene_output, attn), _ = model(videos)
            loss = criterion(action_output, action_targets)

        scene_acc1, scene_acc5 = accuracy(scene_output, scene_targets, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['scene_acc1'].update(scene_acc1.item(), n=batch_size)
        metric_logger.meters['scene_acc5'].update(scene_acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Scene Acc@1 {scene_top1.global_avg:.3f} Scene Acc@5 {scene_top5.global_avg:.3f}  | loss {losses.global_avg:.3f}'
          .format(scene_top1=metric_logger.scene_acc1, scene_top5=metric_logger.scene_acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}