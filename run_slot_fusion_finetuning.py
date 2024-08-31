import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json,math
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment

from datasets import build_dataset,knn_build_dataset
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import  multiple_samples_collate
import utils
import modeling_disentangle
import modeling_finetune
import modeling_slot
import modeling_slot_fusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from kinetics import VideoClsDataset, VideoMAE
from einops import reduce
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_requires_grad_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
class ReturnIndexVideoClsDataset(VideoClsDataset):
    def __getitem__(self, idx):
        # 원래의 VideoClsDataset의 __getitem__ 메서드를 호출
        data = super(ReturnIndexVideoClsDataset, self).__getitem__(idx)
        
        # train 모드에 따라 다르게 반환
        if self.mode == 'train':
            buffer, label, _, index = data
            return buffer, label, index
        
        elif self.mode == 'validation':
            buffer, label, name, index = data
            return buffer, label, index

        else:
            raise NameError('mode {} unkown'.format(self.mode))


@torch.no_grad()
def extract_features(model,scene_model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    action_features = None
    scene_features = None
    scene_targets = None
    
    for batch in metric_logger.log_every(data_loader, 100):
        samples = batch[0]  # batch : (data, label, index)
        index = batch[-1]
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feature,x  = model(samples)
            action_feats = feature
            scene_feats = feature
            action_feats = action_feats.clone()
            scene_feats = scene_feats.clone()
            with torch.no_grad():
                _,teacher_scene_logit = scene_model(samples,return_attn=False)
                scene_target = torch.argmax(teacher_scene_logit, dim=1).float()
                scene_target = scene_target.clone()
        # init storage feature matrix
        if dist.get_rank() == 0 and action_features is None and scene_features is None and scene_targets is None:
            action_features = torch.zeros(len(data_loader.dataset), action_feats.shape[-1])
            scene_features = torch.zeros(len(data_loader.dataset), scene_feats.shape[-1])
            scene_targets = torch.zeros(len(data_loader.dataset))

            action_features = action_features.cuda(non_blocking=True)
            scene_features = scene_features.cuda(non_blocking=True)
            scene_targets = scene_targets.cuda(non_blocking=True)
            print(f"Storing action features into tensor of shape {action_features.shape}")
            print(f"Storing scene features into tensor of shape {scene_features.shape}")
            print(f"Storing scene targets into tensor of shape {scene_targets.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        action_feats_all = torch.empty(
            dist.get_world_size(),
            action_feats.size(0),
            action_feats.size(1),
            dtype=action_feats.dtype,
            device=action_feats.device,
        )

        scene_feats_all = torch.empty(
            dist.get_world_size(),
            scene_feats.size(0),
            scene_feats.size(1),
            dtype=scene_feats.dtype,
            device=scene_feats.device,
        )

        scene_targets_all = torch.empty(
            dist.get_world_size(),
            scene_target.size(0),
            dtype=scene_feats.dtype,
            device=scene_feats.device,
        )


        action_output_l = list(action_feats_all.unbind(0))
        action_output_all_reduce = torch.distributed.all_gather(action_output_l, action_feats, async_op=True)
        action_output_all_reduce.wait()


        scene_output_l = list(scene_feats_all.unbind(0))
        scene_output_all_reduce = torch.distributed.all_gather(scene_output_l, scene_feats, async_op=True)
        scene_output_all_reduce.wait()

        scene_target_l = list(scene_targets_all.unbind(0))
        scene_target_all_reduce = torch.distributed.all_gather(scene_target_l, scene_target, async_op=True)
        scene_target_all_reduce.wait()
        # update storage feature matrix
        if dist.get_rank() == 0:
            action_features.index_copy_(0, index_all, torch.cat(action_output_l))
            scene_features.index_copy_(0, index_all, torch.cat(scene_output_l))
            scene_targets.index_copy_(0, index_all, torch.cat(scene_target_l))
    return action_features,scene_features,scene_targets


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5




def run_knn(model,scene_model,args):
    model.eval()
    scene_model.eval()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    for data_set,data_path in zip(['HMDB51','UCF101'],['filelist/hmdb51','filelist/ucf101']):
        args.data_set =data_set
        args.data_path =data_path
        print(f'KNN {data_set} Start')

        dataset_train, args.nb_classes = knn_build_dataset(is_train=True,  args=args)
        dataset_val, _ = knn_build_dataset(is_train=False, args=args)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_val = %s" % str(sampler_val))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=32,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=32,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_action_features,train_scene_features,train_scene_targets = extract_features(model,scene_model, data_loader_train)
        print("Extracting features for val set...")
        test_action_features,test_scene_features,test_scene_targets = extract_features(model,scene_model, data_loader_val)

        if utils.get_rank() == 0:
            train_action_features = nn.functional.normalize(train_action_features, dim=1, p=2)
            train_scene_features = nn.functional.normalize(train_scene_features, dim=1, p=2)
            test_action_features = nn.functional.normalize(test_action_features, dim=1, p=2)
            test_scene_features = nn.functional.normalize(test_scene_features, dim=1, p=2)

            train_labels = torch.tensor(dataset_train.label_array).long()
            test_labels = torch.tensor(dataset_val.label_array).long()
            
            args.temperature = 0.07
            train_action_features = train_action_features.cuda()
            train_scene_features = train_scene_features.cuda()
            test_action_features = test_action_features.cuda()
            test_scene_features = test_scene_features.cuda()
            train_action_labels = train_labels.cuda()
            test_action_labels = test_labels.cuda()
            train_scene_labels = train_scene_targets.cuda().long()
            test_scene_labels = test_scene_targets.cuda().long()

            print("Features are ready!\nStart the k-NN classification.")
            print('='*20)
            print("train feat : action | train label : action || test feat : action | test label : action")
            for k in args.nb_knn:
                top1, top5 = knn_classifier(train_action_features, train_action_labels,
                    test_action_features, test_action_labels, k, args.temperature)
                print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
            print('='*20)

            print("train feat : scene | train label : scene || test feat : scene | test label : scene")
            
            for k in args.nb_knn:
                top1, top5 = knn_classifier(train_scene_features, train_scene_labels,
                    test_scene_features, test_scene_labels, k, args.temperature)
                print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
            print('='*20)

            print("train feat : action | train label : action || test feat : scene | test label : action")
            for k in args.nb_knn:
                top1, top5 = knn_classifier(train_action_features, train_action_labels,
                    test_scene_features, test_action_labels, k, args.temperature)
                print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
            print('='*20)

            print("train feat : scene | train label : scene || test feat : action | test label : scene")
            
            for k in args.nb_knn:
                top1, top5 = knn_classifier(train_scene_features, train_scene_labels,
                    test_action_features, test_scene_labels, k, args.temperature)
                print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")

        del train_action_features, train_scene_features, test_action_features, test_scene_features
        torch.cuda.empty_cache()
        dist.barrier()




def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--slot_fusion', default='concat', choices=['backbone','concat','sum','fg_only','bg_only'], type=str)
    parser.add_argument('--head_type', type=str, default= 'linear')
    parser.add_argument('--run_knn', action='store_true', default=False)

    #DISENTANGLE
    parser.add_argument('--disentangle_criterion', default='', choices=['UNIFORM','ADVERSARIAL','GRL'], type=str)
    parser.add_argument('--attn_criterion', default='MSE', choices=['MSE','KL', 'CE'], type=str)
    # adapter 쓸지 adapter 쓰면 adapter말고 vit는 다 freeze임 (aggregation은 제외)
    parser.add_argument('--use_adapter', action='store_true', default=False)
    parser.add_argument('--subset', action='store_true', default=False)
    #knn 할때 사용
    parser.add_argument('--nb_knn', default=[10, 20], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    
    # Aggregation parameters
    parser.add_argument('--num_latents', type=int, default= 4)
    parser.add_argument('--weight_tie_layers', type=str2bool, default=True)
    parser.add_argument('--agg_depth', type=int, default= 4)
    # aggregation lr scale
    parser.add_argument('--agg_block_scale', type=float, default= 0.8)

    
    # Model parameters
    parser.add_argument('--model', default='slot_vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    # aggregation에서 query softmax후 key도 softmax로 norm, defualt는 l1 norm임 
    parser.add_argument('--key_softmax', type=int, default=-1)
    parser.add_argument('--no_label', action='store_true', default=False)
    #scene, action head type
    parser.add_argument('--fusion', type=str, default= 'hard_select',choices=['hard_select','attention', 'weightedsum','matching','matching_hard'])
    # 어느 block 까지 얼릴거냐 11이면 all freeze, -1이면 full ft
    parser.add_argument('--grad_from', type=int, default= 8)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--slicing', action='store_true', default=False)
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--data_prefix', default='', type=str)
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_set', default='Kinetics-400', choices=['SCUBA', 'SCUFO', 'SUN397', 'HAT', 'Diving-48','Kinetics-400', 'SSV2', 'UCF101', 'HMDB51','image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--hat_split', default='1', choices=['1', '2', '3'], type=str)
    parser.add_argument('--hat_eval', action='store_true', help='test on HAT three splits at once')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None



    # dataset_scuba_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    # sampler_scuba_val = torch.utils.data.DistributedSampler(
    #     dataset_scuba_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    # collate_func = None
    # data_loader_scuba_val = torch.utils.data.DataLoader(
    #         dataset_scuba_val, sampler=sampler_scuba_val,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False
    #     )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    if 'k400' in args.finetune:
        num_classes= 400
    else:
        num_classes = 101
    
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        init_scale=args.init_scale,
        num_latents=args.num_latents,
        residual = args.residual,
        head_type=args.head_type,
        fusion=args.fusion,
        use_adapter=args.use_adapter,
        key_softmax=args.key_softmax,
        disentangle_criterion=args.disentangle_criterion,
        no_label = args.no_label,
        weight_tie_layers = args.weight_tie_layers,
        agg_depth= args.agg_depth,
        slot_fusion=args.slot_fusion,
        downstream_num_classes = args.nb_classes
    )
    
    
    print("Turning parameters")
    print_requires_grad_parameters(model)



    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding 
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
            agg_block_scale = args.agg_block_scale
            )
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )
        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq

        
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    train_criterion = criterion

    # print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.hat_eval :
        args.data_set = 'HAT'  
        hat_output_dir = os.path.join(current_dir, args.output_dir, 'hat')
        if not os.path.exists(hat_output_dir) :
            os.makedirs(hat_output_dir, exist_ok=True)
        for split in ['1', '2', '3'] :
            # model.module.reset_select_slot_info()
            output_dir = os.path.join(hat_output_dir, split)
            if not os.path.exists(output_dir) :
                os.makedirs(output_dir, exist_ok=True)

            args.hat_split = split
            args.output_dir = output_dir
            args.log_dir = output_dir
            
            dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            if global_rank == 0 and args.log_dir is not None:
                os.makedirs(args.log_dir, exist_ok=True)
                log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
            else:
                log_writer = None

            collate_func = None
            data_loader_test = torch.utils.data.DataLoader(
                    dataset_test, sampler=sampler_test,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False
                )
                    
            preds_file = os.path.join(output_dir, str(global_rank) + '.txt')
            test_stats = final_test(data_loader_test, model, device, preds_file)
            torch.distributed.barrier()
            if global_rank == 0:
                print("Start merging results...")
                final_top1 ,final_top5 = merge(output_dir, num_tasks)
                print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
                model.module.get_select_slot_info()
                log_stats = {'Final top-1': final_top1,
                            'Final Top-5': final_top5}
                if output_dir and utils.is_main_process():
                    with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
                        
        torch.distributed.barrier()
        if global_rank == 0:
            def count_hat_acc(dir, split_dir) :
                accs = []
                for split in split_dir :
                    with open(os.path.join(dir, split, "log.txt"), 'r') as f :
                        data = f.read()
                        data = json.loads(data.replace('\n', ''))
                        accs.append(data["Final top-1"])
                acc = 0
                for a in accs :
                    acc += float(a)
                acc = acc / len(accs)
                print(f"HAT mean acc : {acc}")
            count_hat_acc(dir=hat_output_dir, split_dir=['1', '2', '3'])
        exit(0)

    if args.eval:

        # test_stats = validation_one_epoch(data_loader_val, model, device,args)
        # print(test_stats)

        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = final_test(data_loader_test, model, device, preds_file)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1,
                        'Final Top-5': final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    
        # model = model.float()
        # scene_model = scene_model.float()
        # run_knn(model,scene_model,args)
        exit(0)
    if args.run_knn:

        scene_model =  create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=365,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_checkpoint=args.use_checkpoint,
            use_mean_pooling=False,
            init_scale=args.init_scale,
        )
        scene_model.to(device)


        scene_path = '/data/bkh178/pretrain/checkpoint-best.pth'
        if not os.path.isfile(scene_path):
            scene_path = '/data/gyeongho/pretrain/checkpoint-best.pth'
        weight = torch.load(scene_path, map_location='cpu')['model']

        msg = scene_model.load_state_dict(weight)
        print(f'scene model load weight msg : {msg}')
        
        for param in scene_model.parameters():
            param.requires_grad = False

        model = model.float()
        scene_model = scene_model.float()
        run_knn(model,scene_model,args)
        exit(0)
        
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = validation_one_epoch(data_loader_val, model, device)
            # print('SCUBA Validation')
            # validation_one_epoch(data_loader_scuba_val, model, device,args)
            print(f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(val_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(val_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
            
            
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(data_loader_test, model, device, preds_file)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1,
                    'Final Top-5': final_top5}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    torch.distributed.barrier()


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
