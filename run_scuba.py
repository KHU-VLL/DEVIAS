import torch
import utils
from datasets import build_dataset
import os
import json


def run_scuba(model, args, test_func, merge_func, test_scene_func=None, scene_model=None):
    # #! SCUBA fg & bg test
    if args.data_set == "Kinetics-400" :
        args.data_path = os.path.join(os.getcwd(), 'filelist/scuba/kinetics')
    elif args.data_set == "UCF101" :
        args.data_path = os.path.join(os.getcwd(), 'filelist/scuba/ucf101')
    
    args.data_set = 'SCUBA'  
    print(f"Dataset path : {args.data_prefix}")
    
    #! fix test view
    args.test_num_segment, args.test_num_crop = 2, 3
    
    model.eval()
    if scene_model is not None :
        scene_model.eval()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    device = model.device

    dataset_scuba_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    sampler_scuba_test = torch.utils.data.DistributedSampler(
        dataset_scuba_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    collate_func = None
    data_loader_scuba_test = torch.utils.data.DataLoader(
            dataset_scuba_test, sampler=sampler_scuba_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    
    #! SCUBA fg test
    print('#'*20+'\nFG TEST\n'+'#'*20)
    # model.module.reset_select_slot_info()
    scuba_fg_output_dir = os.path.join(args.output_dir, 'scuba', 'fg')
    if not os.path.exists(scuba_fg_output_dir) :
        os.makedirs(scuba_fg_output_dir, exist_ok=True)
    
    preds_file = os.path.join(scuba_fg_output_dir, str(global_rank) + '.txt')
    test_stats = test_func(data_loader_scuba_test, model, device, preds_file)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1, final_top5 = merge_func(scuba_fg_output_dir, num_tasks)
        print(f"Accuracy of the network on the {len(dataset_scuba_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1,
                    'Final Top-5': final_top5}
        if scuba_fg_output_dir and utils.is_main_process():
            with open(os.path.join(scuba_fg_output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # ! SCUBA bg test
    if test_scene_func is not None :
        print('#'*20+'\nBG TEST\n'+'#'*20)
        scuba_bg_output_dir = os.path.join(args.output_dir, 'scuba', 'bg')
        if not os.path.exists(scuba_bg_output_dir) :
            os.makedirs(scuba_bg_output_dir, exist_ok=True)
        
        preds_file = os.path.join(scuba_bg_output_dir, str(global_rank) + '.txt')
        test_stats = test_scene_func(data_loader_scuba_test, model, scene_model, device, preds_file, num_labels=args.nb_classes)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1, final_top5 = merge_func(scuba_bg_output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_scuba_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1,
                        'Final Top-5': final_top5}
            if scuba_bg_output_dir and utils.is_main_process():
                with open(os.path.join(scuba_bg_output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")