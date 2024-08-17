import torch
import os
import torch.distributed as dist

from datasets import knn_build_dataset
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import torch.nn as nn
from kinetics import VideoClsDataset, VideoMAE


class ReturnIndexVideoClsDataset(VideoClsDataset):
    def __getitem__(self, idx):
        # call __getitem__ from original VideoClsDataset
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
            #! for slot model
            (fg_feat, bg_feat), (fg_logit, bg_logit, _),(slots_head, slots)  = model(samples)
            #! for disentangle model
            (fg_feat, action_logit), (bg_feat, scene_logit) = model(samples)
        
            action_feats = fg_feat
            scene_feats = bg_feat
            action_feats = action_feats.clone()
            scene_feats = scene_feats.clone()
            with torch.no_grad():
                _, teacher_scene_logit = scene_model(samples,return_attn=False)
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
    for data_set,data_path in zip(['HMDB51','UCF101','Diving-48'],['filelist/hmdb51','filelist/ucf101','filelist/diving48']):
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


            torch.save(train_action_features.cpu(), os.path.join(args.output_dir, f"{data_set}_train_action_features.pth"))
            torch.save(train_scene_features.cpu(), os.path.join(args.output_dir, f"{data_set}_train_scene_features.pth"))
            torch.save(test_action_features.cpu(), os.path.join(args.output_dir, f"{data_set}_test_action_features.pth"))
            torch.save(test_scene_features.cpu(), os.path.join(args.output_dir, f"{data_set}_test_scene_features.pth"))
            torch.save(train_action_labels.cpu(), os.path.join(args.output_dir, f"{data_set}_train_action_labels.pth"))
            torch.save(test_action_labels.cpu(), os.path.join(args.output_dir, f"{data_set}_test_action_labels.pth"))
            torch.save(train_scene_labels.cpu(), os.path.join(args.output_dir, f"{data_set}_train_scene_labels.pth"))
            torch.save(test_scene_labels.cpu(), os.path.join(args.output_dir, f"{data_set}_test_scene_labels.pth"))

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