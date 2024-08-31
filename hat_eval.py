import torch
import os
from datasets import build_dataset
import utils
import json


def hat_eval(args, model, test_func, merge_func, scene_model=None) :
    anno_path = args.hat_anno_path  # like filelist/hat/ucf101/rand
    if 'kinetics' in anno_path :
        args.data_set = 'Kinetics-HAT'
    elif 'ucf101' in anno_path :
        args.data_set = 'UCF101-HAT'
    else :
        raise NotImplementedError()
            
    current_dir = os.getcwd()
    hat_output_dir = os.path.join(current_dir, args.output_dir, 'hat')
    if not os.path.exists(hat_output_dir) :
        os.makedirs(hat_output_dir, exist_ok=True)
        
    for split in ['1', '2', '3'] :  
        # model.module.reset_select_slot_info()
        hat_ver = anno_path.split("/")[-1]
        assert hat_ver in ['rand', 'far', 'close']
        args.data_path = os.path.join(anno_path, f"actionswap_{hat_ver}_{split}.pickle")
        
        output_dir = os.path.join(hat_output_dir, hat_ver, split)
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
        device = model.device
        
        if scene_model is not None :
            test_stats = test_func(data_loader_test, model, scene_model, device, preds_file, num_labels=args.nb_classes)
        else :
            test_stats = test_func(data_loader_test, model, device, preds_file)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1, final_top5 = merge_func(output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            # model.module.get_select_slot_info()
            log_stats = {'Final top-1': final_top1,
                        'Final Top-5': final_top5}
            if output_dir and utils.is_main_process():
                with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
    
    torch.distributed.barrier()
    if global_rank == 0:
        from count_hat_acc import count_hat_acc
        print(count_hat_acc(dir=os.path.join(hat_output_dir, hat_ver), split_dir=['1', '2', '3'], topk=1))
